"""Exercise the same HTTP routes as the UI with a real DAVIS clip."""

import argparse
import io
import json
import zipfile

import av
import numpy as np
from PIL import Image

from sam_app.models import ROOT
from training.data import DATA, prompt, scores


def check(r):
    if r.status_code != 200:
        raise RuntimeError(f"{r.request.url}: {r.status_code} {r.text[:250]}")
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--url", help="已运行的本地应用地址")
    args = parser.parse_args()
    dest = ROOT / "outputs/demo"
    dest.mkdir(exist_ok=True, parents=True)
    frames = sorted((DATA / "JPEGImages/480p/car-turn").glob("*.jpg"))
    with Image.open(
        DATA / "Annotations/480p/car-turn" / (frames[0].stem + ".png")
    ) as im:
        ids = np.array(im)
    obj = int(np.setdiff1d(np.unique(ids), [0, 255])[0])
    truth = ids == obj
    pts, lab, _ = prompt(truth, np.random.default_rng(7), "point")
    _, _, box = prompt(truth, np.random.default_rng(7), "box", 0)
    report = {}
    if args.url:
        import httpx

        connection = httpx.Client(base_url=args.url, timeout=360, trust_env=False)
    else:
        from fastapi.testclient import TestClient

        from sam_app.server import app

        connection = TestClient(app)
    with connection as client:
        report["device"] = check(client.get("/api/status")).json()
        with frames[0].open("rb") as file:
            spec = check(
                client.post(
                    "/api/open", files={"file": ("car.jpg", file, "image/jpeg")}
                )
            ).json()
        base = "/api/" + spec["id"]
        paths = [("point", {"points": pts, "labels": lab}), ("box", {"box": box})]
        if args.text:
            options = check(client.post(base + "/detect", json={"text": "汽车"})).json()
            if not options:
                raise AssertionError("示例汽车未检出")
            paths.append(("text", {"box": options[0]["box"]}))
            report["text_candidates"] = len(options)
        for name, p in paths:
            res = check(client.post(base + "/segment", json={"prompt": p})).json()
            prefix = base + "/result/" + res["revision"]
            mask = (
                np.array(
                    Image.open(io.BytesIO(check(client.get(prefix + "/mask")).content))
                )
                > 0
            )
            data = check(client.get(prefix + "/cutout")).content
            rgba = np.array(Image.open(io.BytesIO(data)))
            assert np.array_equal(rgba[:, :, 3] > 0, mask) and mask.shape == truth.shape
            report[name + "_iou"] = scores(mask, truth, ids != 255)[0]
            (dest / (name + "-foreground.png")).write_bytes(data)
        if args.video:
            clip = dest / "car-turn.mp4"
            with av.open(str(clip), "w") as out:
                stream = out.add_stream("libx264", rate=8)
                stream.width = ids.shape[1]
                stream.height = ids.shape[0]
                stream.pix_fmt = "yuv420p"
                for f in frames:
                    with Image.open(f) as im:
                        vf = av.VideoFrame.from_image(im.convert("RGB"))
                    for packet in stream.encode(vf):
                        out.mux(packet)
                for packet in stream.encode():
                    out.mux(packet)
            with clip.open("rb") as f:
                vid = check(
                    client.post(
                        "/api/open", files={"file": ("car-turn.mp4", f, "video/mp4")}
                    )
                ).json()
            assert vid["count"] == len(frames)
            prefix = "/api/" + vid["id"]
            keyframes = {"0": {"box": box}}
            mid = len(frames) // 2
            for mode in ("initial", "corrected"):
                if mode == "corrected":
                    with Image.open(
                        DATA / "Annotations/480p/car-turn" / (frames[mid].stem + ".png")
                    ) as im:
                        ids_mid = np.array(im)
                    pts_mid, lab_mid, _ = prompt(
                        ids_mid == obj, np.random.default_rng(12), "point"
                    )
                    keyframes[str(mid)] = {"points": pts_mid, "labels": lab_mid}
                res = check(
                    client.post(prefix + "/propagate", json={"keyframes": keyframes})
                ).json()
                key = res["export"]
                tracked = check(client.get(prefix + f"/tracked/{key}/{mid}"))
                assert Image.open(io.BytesIO(tracked.content)).size == (
                    vid["w"],
                    vid["h"],
                )
            for file in ("preview.mp4", "foreground.zip", "prompts.json"):
                (dest / file).write_bytes(
                    check(client.get(prefix + "/export/" + key + "/" + file)).content
                )
            with zipfile.ZipFile(dest / "foreground.zip") as archive:
                assert len(archive.namelist()) == len(frames)
                with av.open(str(clip)) as src:
                    first_rgb = next(src.decode(video=0)).to_ndarray(format="rgb24")
                with archive.open("000000.png") as png_file:
                    exported_rgb = np.array(Image.open(png_file))[:, :, :3]
                assert np.array_equal(first_rgb, exported_rgb)
            with av.open(str(dest / "preview.mp4")) as out:
                assert sum(1 for _ in out.decode(video=0)) == len(frames)
            report["corrected_video_frames"] = len(frames)
    (dest / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
