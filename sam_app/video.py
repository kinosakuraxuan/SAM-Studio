import zipfile
from fractions import Fraction

import av
import cv2
import numpy as np
from PIL import Image

from .models import make_model
from .styles import stylize


def decode(path, dest):
    dest.mkdir()
    times = []
    with av.open(str(path)) as src:
        stream = src.streams.video[0]
        rate = stream.average_rate or Fraction(25)
        for idx, frame in enumerate(src.decode(stream)):
            timestamp = (
                float(frame.time) if frame.time is not None else idx / float(rate)
            )
            times.append(timestamp)
            frame.to_image().convert("RGB").save(dest / f"{idx:06d}.jpg", quality=97)
    if not times:
        raise ValueError("视频没有可解码的帧")
    return times, rate


def write_browser_preview(path, destination):
    """Create a seekable H.264 copy for browsers that cannot play the source."""
    with (
        av.open(str(path)) as original,
        av.open(str(destination), mode="w", options={"movflags": "+faststart"}) as output,
    ):
        stream = original.streams.video[0]
        rate = stream.average_rate or Fraction(25)
        scale = min(1.0, 1280 / max(stream.width, stream.height))
        encoder = output.add_stream("libx264", rate=rate)
        encoder.width = max(2, round(stream.width * scale / 2) * 2)
        encoder.height = max(2, round(stream.height * scale / 2) * 2)
        encoder.pix_fmt = "yuv420p"
        encoder.options = {"preset": "veryfast", "crf": "23"}
        clock = Fraction(1, 90000)
        encoder.time_base = encoder.codec_context.time_base = clock
        first = None
        previous_pts = -1
        for index, frame in enumerate(original.decode(stream)):
            timestamp = frame.time if frame.time is not None else index / float(rate)
            if first is None:
                first = timestamp
            pts = max(previous_pts + 1, round((timestamp - first) / clock))
            image = frame.reformat(
                width=encoder.width, height=encoder.height, format=encoder.pix_fmt
            )
            image.pts = previous_pts = pts
            image.time_base = clock
            for packet in encoder.encode(image):
                output.mux(packet)
        for packet in encoder.encode():
            output.mux(packet)


def track_frames(frames, count, prompts, engine, *, offload_state=False):
    if 0 not in prompts:
        raise ValueError("请先在首帧指定目标。")
    state = engine.init_state(
        video_path=str(frames),
        offload_video_to_cpu=True,
        offload_state_to_cpu=offload_state,
    )
    keys = sorted(prompts)
    for pos, k in enumerate(keys):
        p = prompts[k]
        engine.add_new_points_or_box(
            state,
            frame_idx=k,
            obj_id=1,
            points=p.points or None,
            labels=p.labels or None,
            box=p.box,
            clear_old_points=True,
        )
        correction = keys[pos + 1] if pos + 1 < len(keys) else None
        end = count - 1 if correction is None else correction
        for i, ids, logits in engine.propagate_in_video(
            state, start_frame_idx=k, max_frame_num_to_track=end - k
        ):
            # Publish the corrected mask when the next prompt falls on this frame.
            if i != correction:
                mask = (logits[ids.index(1), 0] > 0).cpu().numpy()
                yield i, np.packbits(mask)


def track(frames, count, prompts, engine):
    masks = dict(track_frames(frames, count, prompts, engine))
    if len(masks) != count:
        raise RuntimeError("掩膜帧数与源视频不一致")
    return masks


def render(frames, source, times, rate, prompts, version, out):
    masks = track(frames, len(times), prompts, make_model(version, video=True))
    write_video(source, times, rate, masks, out)
    return masks


def style_windows(masks, width, height):
    bounds = []
    for index in range(len(masks)):
        bits = np.unpackbits(masks[index], count=width * height).reshape(height, width)
        x, y, w, h = cv2.boundingRect(bits)
        bounds.append((x, y, x + w, y + h) if w and h else None)
    windows = []
    for index, box in enumerate(bounds):
        if box is None:
            windows.append(None)
            continue
        neighbors = [b for b in bounds[max(0, index - 2) : index + 3] if b]
        left, top, right, bottom = np.median(neighbors, axis=0).round().astype(int)
        x, y = min(left, box[0]), min(top, box[1])
        windows.append((x, y, max(right, box[2]) - x, max(bottom, box[3]) - y))
    return windows


def highlight_mask(rgb, mask):
    rgb[mask] = rgb[mask] * 0.53 + np.array([46, 211, 157]) * 0.47


def write_video(
    source,
    times,
    rate,
    masks,
    out,
    style=None,
    strength=100,
    windows=None,
):
    preview = out / ("styled.mp4" if style else "preview.mp4")
    archive = out / ("styled-frames.zip" if style else "foreground.zip")
    with (
        av.open(str(source)) as original,
        av.open(str(preview), mode="w") as output,
        zipfile.ZipFile(archive, "w") as z,
    ):
        enc = output.add_stream("libx264", rate=rate)
        stream = original.streams.video[0]
        w, h = stream.width, stream.height
        enc.width, enc.height = w, h
        enc.pix_fmt = "yuv444p" if w % 2 or h % 2 else "yuv420p"
        enc.time_base = Fraction(1, 90000)
        enc.codec_context.time_base = enc.time_base
        durations = {}

        def mux(packets):
            for packet in packets:
                if packet.pts in durations:
                    packet.duration = durations.pop(packet.pts)
                output.mux(packet)

        for i, source_frame in enumerate(original.decode(video=0)):
            t = times[i]
            rgb = source_frame.to_ndarray(format="rgb24")
            mask = np.unpackbits(masks[i], count=w * h).reshape(h, w).astype(bool)
            if style:
                rgb = stylize(
                    rgb, mask, style, strength, windows[i] if windows else None
                )
                saved = Image.fromarray(rgb)
            else:
                saved = Image.fromarray(rgb).convert("RGBA")
                saved.putalpha(Image.fromarray(mask.astype(np.uint8) * 255))
            with z.open(f"{i:06d}.png", "w") as target:
                saved.save(target, format="PNG")
            if not style:
                highlight_mask(rgb, mask)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.time_base = Fraction(1, 90000)
            frame.pts = round((t - times[0]) * 90000)
            if i + 1 < len(times):
                end = times[i + 1]
            else:
                last_duration = (
                    float(source_frame.duration * source_frame.time_base)
                    if source_frame.duration and source_frame.time_base
                    else 1 / float(rate)
                )
                end = t + last_duration
            durations[frame.pts] = max(1, round((end - times[0]) * 90000) - frame.pts)
            frame.duration = durations[frame.pts]
            mux(enc.encode(frame))
        mux(enc.encode())
