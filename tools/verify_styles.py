"""Check the interactive style endpoints with real SAM masks and decoded video frames."""

import io
import json
from zipfile import ZipFile

import av
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from sam_app.server import WEB, app
from sam_app.styles import stylize
from training.data import DATA, ROOT

STYLES = ("neural-cartoon", "neural-sketch")


def response(client, method, route, **kwargs):
    result = client.request(method, route, **kwargs)
    result.raise_for_status()
    return result


def rgb_png(data):
    with Image.open(io.BytesIO(data)) as im:
        return np.array(im)


def check_still(client, dest, seq, name):
    path = DATA / "JPEGImages/480p" / seq / "00000.jpg"
    with Image.open(path) as im:
        original = np.array(im.convert("RGB"))
    with Image.open(DATA / "Annotations/480p" / seq / "00000.png") as im:
        ids = np.array(im)
    values, counts = np.unique(ids[(ids != 0) & (ids != 255)], return_counts=True)
    y, x = np.where(ids == values[np.argmax(counts)])
    box = [int(x.min()), int(y.min()), int(x.max()), int(y.max())]
    with path.open("rb") as f:
        sid = response(
            client, "POST", "/api/open", files={"file": (path.name, f, "image/jpeg")}
        ).json()["id"]
    base = f"/api/{sid}"
    key = response(
        client, "POST", base + "/segment", json={"prompt": {"box": box}}
    ).json()["revision"]
    mask = rgb_png(response(client, "GET", base + f"/result/{key}/mask").content) > 0
    assert mask.any() and (~mask).any()
    Image.fromarray(original).save(dest / f"{name}-original.png")
    changed = {}
    for style in STYLES:
        path = base + f"/style/{key}/0/{style}/"
        for level in (0, 50, 100):
            raw = response(
                client, "GET", path + "image", params={"strength": level}
            ).content
            result = rgb_png(raw)
            assert np.array_equal(result[~mask], original[~mask])
            if level == 0:
                assert np.array_equal(result, original)
            else:
                changed[f"{style}-{level}"] = int(
                    np.any(result[mask] != original[mask], axis=1).sum()
                )
                assert changed[f"{style}-{level}"] > 0
            if level == 100:
                (dest / f"{name}-{style}.png").write_bytes(raw)
        cutout = rgb_png(response(client, "GET", path + "cutout").content)
        assert np.array_equal(cutout[:, :, 3], mask.astype(np.uint8) * 255)
        assert np.array_equal(cutout[:, :, :3], result)
    return {"changed_foreground_pixels": changed, "changed_background_pixels": 0}


def check_edges():
    rng = np.random.default_rng(9)
    original = rng.integers(0, 256, (29, 31, 3), dtype=np.uint8)
    ring = np.zeros(original.shape[:2], bool)
    ring[:22, :23] = True
    ring[3:13, 5:12] = False
    for mask in (ring, np.zeros_like(ring), np.ones_like(ring), ring[:1, :1]):
        cropped = original[: mask.shape[0], : mask.shape[1]]
        for style in STYLES:
            out = stylize(cropped, mask, style)
            assert np.array_equal(out[~mask], cropped[~mask])
            assert np.array_equal(stylize(cropped, mask, style, 0), cropped)


def check_alpha(client):
    with Image.open(WEB / "assets/camel-cutout.webp") as im:
        original = np.array(im.convert("RGBA"))
    original[::4, ::4, 3] //= 2
    buf = io.BytesIO()
    Image.fromarray(original).save(buf, format="PNG")
    sid = response(
        client,
        "POST",
        "/api/open",
        files={"file": ("alpha.png", buf.getvalue(), "image/png")},
    ).json()["id"]
    y, x = np.where(original[:, :, 3] > 0)
    box = np.array([x.min(), y.min(), x.max(), y.max()]).tolist()
    base = f"/api/{sid}"
    key = response(
        client, "POST", base + "/segment", json={"prompt": {"box": box}}
    ).json()["revision"]
    mask = rgb_png(response(client, "GET", base + f"/result/{key}/mask").content) > 0
    unchanged = ~mask | (original[:, :, 3] == 0)
    for style in STYLES:
        path = base + f"/style/{key}/0/{style}/"
        composite = rgb_png(response(client, "GET", path + "image").content)
        cutout = rgb_png(response(client, "GET", path + "cutout").content)
        assert np.array_equal(composite[:, :, 3], original[:, :, 3])
        assert np.array_equal(composite[unchanged], original[unchanged])
        assert np.array_equal(cutout[:, :, 3], np.where(mask, original[:, :, 3], 0))


def check_video(client, dest):
    clip = WEB / "assets/car-turn.mp4"
    with clip.open("rb") as f:
        vid = response(
            client, "POST", "/api/open", files={"file": (clip.name, f, "video/mp4")}
        ).json()
    base = "/api/" + vid["id"]
    export = response(
        client,
        "POST",
        base + "/propagate",
        json={"keyframes": {"0": {"box": [700, 212, 800, 304]}}},
    ).json()["export"]
    masks = response(client, "GET", base + f"/export/{export}/foreground.zip").content
    checks = {}
    for style in STYLES:
        response(
            client,
            "POST",
            base + f"/video-style/{export}",
            json={"style": style, "strength": 100},
        )
        prefix = base + f"/video-style/{export}/{style}/100/"
        packed = response(client, "GET", prefix + "styled-frames.zip").content
        mp4 = dest / (style + ".mp4")
        mp4.write_bytes(response(client, "GET", prefix + "styled.mp4").content)
        frames_with_styled_pixels = 0
        with (
            av.open(str(clip)) as src,
            av.open(str(mp4)) as preview,
            ZipFile(io.BytesIO(masks)) as mz,
            ZipFile(io.BytesIO(packed)) as archive,
        ):
            frame_count = 0
            assert len(archive.namelist()) == vid["count"]
            for i, (frame, rendered) in enumerate(
                zip(src.decode(video=0), preview.decode(video=0), strict=True)
            ):
                rgb = frame.to_ndarray(format="rgb24")
                name = f"{i:06d}.png"
                mask = rgb_png(mz.read(name))[:, :, 3] > 0
                result = rgb_png(archive.read(name))
                assert np.array_equal(result[~mask], rgb[~mask])
                assert abs(frame.time - rendered.time) < 0.001
                assert (rendered.width, rendered.height) == (frame.width, frame.height)
                frames_with_styled_pixels += int(np.any(result[mask] != rgb[mask]))
                frame_count += 1
            assert frame_count == vid["count"] and frames_with_styled_pixels > 0
        checks[style] = {
            "frames": frame_count,
            "changed_background_pixels": 0,
            "frames_with_styled_pixels": frames_with_styled_pixels,
        }
    return checks


if __name__ == "__main__":
    destination = ROOT / "outputs/neural-style-demo"
    destination.mkdir(parents=True, exist_ok=True)
    check_edges()
    with TestClient(app) as client:
        report = {
            label: check_still(client, destination, seq, label)
            for seq, label in (("classic-car", "car"), ("dance-twirl", "person"))
        }
        check_alpha(client)
        report["alpha_preserved"] = True
        report["video"] = check_video(client, destination)
    (destination / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
