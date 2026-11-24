import io
import json
import shutil
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import av
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
from pydantic import BaseModel, Field

from . import models, video
from .styles import Style, stylize

LOCK = threading.RLock()
current = {}
WEB = Path(__file__).with_name("web")


def release():
    if current:
        folder = current["temp"]
        current.clear()
        folder.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@asynccontextmanager
async def lifespan(app):
    yield
    with LOCK:
        release()


app = FastAPI(title="本地交互分割", lifespan=lifespan)


class Prompt(BaseModel):
    points: list[tuple[float, float]] = Field(default_factory=list)
    labels: list[int] = Field(default_factory=list)
    box: tuple[float, float, float, float] | None = None


class Request(BaseModel):
    frame: int = 0
    version: str = "original"
    prompt: Prompt = Field(default_factory=Prompt)
    text: str = ""
    keyframes: dict[int, Prompt] = Field(default_factory=dict)


class StyleSettings(BaseModel):
    style: Style
    strength: int = Field(default=100, ge=0, le=100)


def session(sid, index=0):
    if current.get("id") != sid:
        raise HTTPException(409, "素材已切换，请重新打开。")
    if not 0 <= index < current["count"]:
        raise HTTPException(400, "帧号越界")
    return current


def valid_prompt(p, w, h):
    if len(p.points) != len(p.labels) or any(x not in (0, 1) for x in p.labels):
        raise ValueError("点与标签数量或值不正确")
    for x, y in p.points:
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError("点击超出图像")
    if p.box is not None:
        x, y, r, b = p.box
        if not (0 <= x < r <= w and 0 <= y < b <= h):
            raise ValueError("请在图像内拖出有效矩形")
    if not p.points and p.box is None:
        raise ValueError("请添加提示")


def png(image, metadata=None):
    buf = io.BytesIO()
    details = PngInfo()
    if metadata is not None:
        details.add_itxt("sam", json.dumps(metadata, ensure_ascii=False))
    image.save(buf, format="PNG", pnginfo=details)
    return Response(content=buf.getvalue(), media_type="image/png")


def frame_image(s, i):
    if s["video"] and s.get("rgb_index") != i:
        with av.open(str(s["root"] / "source")) as source:
            for index, frame in enumerate(source.decode(video=0)):
                if index == i:
                    s["rgb"] = frame.to_ndarray(format="rgb24")
                    s["rgb_index"] = i
                    break
    return s["rgb"]


def with_alpha(rgb, alpha=None, mask=None):
    img = Image.fromarray(rgb)
    if mask is not None:
        alpha = np.where(mask, 255 if alpha is None else alpha, 0).astype(np.uint8)
    if alpha is not None:
        img.putalpha(Image.fromarray(alpha))
    return img


@app.exception_handler(ValueError)
@app.exception_handler(FileNotFoundError)
async def bad_input(request, exc):
    from fastapi.responses import JSONResponse

    return JSONResponse({"detail": str(exc)}, status_code=422)


@app.get("/api/status")
def status():
    return {
        "device": models.device(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
        "tuned": models.TUNED.is_file(),
        "refined": models.TUNED.is_file() and models.REFINED.is_file(),
        "dino": (models.DINO / "model.safetensors").is_file(),
        "live_preview": True,
    }


@app.post("/api/open")
def open_media(file: UploadFile):
    folder = tempfile.TemporaryDirectory(prefix="sam-local-")
    root = Path(folder.name)
    isvideo = Path(file.filename or "").suffix.lower() in {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".m4v",
    }
    try:
        info = {
            "video": isvideo,
            "root": root,
            "temp": folder,
            "id": uuid.uuid4().hex,
            "version": None,
            "image_key": None,
            "mask_key": None,
            "export": None,
        }
        if isvideo:
            source = root / "source"
            with source.open("wb") as out:
                shutil.copyfileobj(file.file, out)
            info["times"], info["rate"] = video.decode(source, root / "frames")
            info["count"] = len(info["times"])
            with Image.open(root / "frames/000000.jpg") as im:
                w, h = im.size
        else:
            with Image.open(file.file) as im:
                upright = ImageOps.exif_transpose(im)
                info["rgb"] = np.array(upright.convert("RGB"))
                if "A" in upright.getbands() or "transparency" in upright.info:
                    info["alpha"] = np.array(upright.convert("RGBA").getchannel("A"))
            h, w = info["rgb"].shape[:2]
            info["count"] = 1
        info.update(w=w, h=h)
    except Exception as err:
        folder.cleanup()
        raise HTTPException(400, "无法解码该素材: " + str(err)) from err
    with LOCK:
        release()
        current.update(info)
    return {k: info[k] for k in ("id", "count", "video", "w", "h")}


@app.get("/api/{sid}/frame/{index}")
def get_frame(sid: str, index: int):
    with LOCK:
        s = session(sid, index)
        return png(with_alpha(frame_image(s, index), s.get("alpha")))


@app.get("/api/{sid}/playback/{kind}")
def compatible_playback(
    sid: str,
    kind: Literal["original", "tracked", "styled"],
    token: str = "",
    style: Style | None = None,
    strength: int = Query(default=100, ge=0, le=100),
):
    with LOCK:
        s = session(sid)
        if not s["video"]:
            raise HTTPException(400, "请打开视频")
        if kind == "original":
            source = s["root"] / "source"
        elif token != s["export"]:
            raise HTTPException(409, "跟踪结果已更新")
        elif kind == "tracked":
            source = s["root"] / token / "preview.mp4"
        else:
            source = s["root"] / token / f"{style}-{strength}" / "styled.mp4"
        if not source.is_file():
            raise HTTPException(404, "请先生成要播放的视频")
        preview = source.with_suffix(".browser.mp4")
        if not preview.is_file():
            with tempfile.TemporaryDirectory(dir=preview.parent) as temp:
                staging = Path(temp) / "preview.mp4"
                video.write_browser_preview(source, staging)
                staging.replace(preview)
        return FileResponse(preview, media_type="video/mp4")


@app.post("/api/{sid}/segment")
def segment(sid: str, req: Request):
    with LOCK:
        s = session(sid, req.frame)
        valid_prompt(req.prompt, s["w"], s["h"])
        base_mtime = (
            models.TUNED.stat().st_mtime_ns
            if req.version in ("tuned", "refined") and models.TUNED.is_file()
            else 0
        )
        weights_mtime = (
            models.REFINED.stat().st_mtime_ns
            if req.version == "refined" and models.REFINED.is_file()
            else base_mtime
        )
        signature = (req.version, weights_mtime, base_mtime)
        key = (signature, req.frame)
        if s["version"] != signature:
            s["engine"] = models.predictor(req.version)
            s["version"] = signature
            s["image_key"] = None
        if s["image_key"] != key:
            s["engine"].set_image(frame_image(s, req.frame))
            s["image_key"] = key
        mask = models.mask_from_prompt(
            s["engine"], req.prompt.points, req.prompt.labels, req.prompt.box
        )
        token = uuid.uuid4().hex
        s.update(
            mask=mask,
            mask_key=token,
            mask_frame=req.frame,
            mask_metadata={
                "version": req.version,
                "weights_mtime_ns": signature[1],
                "frame": req.frame,
                "prompt": req.prompt.model_dump(),
            },
        )
        return {"revision": token}


@app.get("/api/{sid}/result/{revision}/{kind}")
def result(sid: str, revision: str, kind: str):
    with LOCK:
        s = session(sid)
        if s["mask_key"] != revision:
            raise HTTPException(409, "请重新分割当前图像")
        mask = Image.fromarray(s["mask"].astype("uint8") * 255)
        if kind == "mask":
            return png(mask, s["mask_metadata"])
        if kind != "cutout":
            raise HTTPException(404)
        img = with_alpha(frame_image(s, s["mask_frame"]), s.get("alpha"), s["mask"])
        return png(img, s["mask_metadata"])


def tracked_mask(s, token, index):
    if s["export"] != token:
        raise HTTPException(409, "跟踪结果已更新")
    return (
        np.unpackbits(s["tracked_masks"][index], count=s["w"] * s["h"])
        .reshape(s["h"], s["w"])
        .astype(bool)
    )


@app.get("/api/{sid}/style/{token}/{index}/{style}/{kind}")
def style_image(
    sid: str,
    token: str,
    index: int,
    style: Style,
    kind: Literal["image", "cutout"],
    strength: int = Query(100, ge=0, le=100),
):
    with LOCK:
        s = session(sid, index)
        if s["mask_key"] == token and s["mask_frame"] == index:
            mask, metadata = s["mask"], s["mask_metadata"]
        elif s["video"] and s["export"] == token:
            mask, metadata = (
                tracked_mask(s, token, index),
                {"frame": index, "track": token},
            )
        else:
            raise HTTPException(409, "目标掩膜已更新，请重新预览")
        key = (token, index, style, strength)
        if s.get("style_key") != key:
            editable = mask if "alpha" not in s else (mask & (s["alpha"] > 0))
            window = s["windows"][index] if token == s["export"] else None
            s["style_rgb"] = stylize(
                frame_image(s, index), editable, style, strength, window
            )
            s["style_key"] = key
        output = with_alpha(
            s["style_rgb"], s.get("alpha"), mask if kind == "cutout" else None
        )
        return png(output, {**metadata, "style": style, "strength": strength})


@app.post("/api/{sid}/detect")
def detect(sid: str, req: Request):
    with LOCK:
        s = session(sid, req.frame)
        return models.candidates(frame_image(s, req.frame), req.text)


def tracking_key(s, req):
    if not s["video"]:
        raise HTTPException(400, "请打开视频")
    if 0 not in req.keyframes:
        raise HTTPException(400, "请先在首帧标记目标。")
    for index, prompt in req.keyframes.items():
        if not 0 <= index < s["count"]:
            raise HTTPException(400, "帧号越界")
        valid_prompt(prompt, s["w"], s["h"])
    return (
        req.version,
        models.TUNED.stat().st_mtime_ns
        if req.version == "tuned" and models.TUNED.is_file()
        else 0,
        tuple((i, p.model_dump_json()) for i, p in sorted(req.keyframes.items())),
    )


def advance_preview(s, job, index):
    masks = job["masks"]
    if index in masks:
        return
    if index != len(masks):
        raise HTTPException(409, "请按顺序处理视频帧。")
    try:
        if job["iterator"] is None:
            job["iterator"] = video.track_frames(
                s["root"] / "frames",
                s["count"],
                job["prompts"],
                models.make_model(job["key"][0], video=True),
                offload_state=True,
            )
        frame, mask = next(job["iterator"])
        if frame != index:
            raise RuntimeError("跟踪帧号与源视频不一致。")
        masks[frame] = mask
        if len(masks) == s["count"]:
            job["iterator"].close()
            job["iterator"] = None
    except (RuntimeError, StopIteration) as err:
        s.pop("live_preview", None)
        raise HTTPException(500, str(err) or "跟踪提前结束，请重试。") from err


@app.post("/api/{sid}/live-preview")
def start_live_preview(sid: str, req: Request):
    with LOCK:
        s = session(sid)
        key = tracking_key(s, req)
        if "live_preview" not in s or s["live_preview"]["key"] != key:
            s["live_preview"] = {
                "id": uuid.uuid4().hex,
                "key": key,
                "prompts": req.keyframes,
                "iterator": None,
                "masks": s["tracked_masks"] if s.get("export_key") == key else {},
            }
        job = s["live_preview"]
        return {
            "id": job["id"],
            "ready": len(job["masks"]),
            "times": s["times"],
            "interval": 1 / float(s["rate"]),
        }


@app.get("/api/{sid}/live-preview/{token}/{index}")
def live_preview_frame(
    sid: str,
    token: str,
    index: int,
    effect: Literal["mask", "neural-cartoon", "neural-sketch"] = "mask",
    strength: int = Query(100, ge=0, le=100),
):
    with LOCK:
        s = session(sid, index)
        job = s.get("live_preview")
        if not job or job["id"] != token:
            raise HTTPException(409, "预览选区已更新，请重新播放。")
        advance_preview(s, job, index)
        mask = (
            np.unpackbits(job["masks"][index], count=s["w"] * s["h"])
            .reshape(s["h"], s["w"])
            .astype(bool)
        )
        with Image.open(s["root"] / "frames" / f"{index:06d}.jpg") as frame:
            rgb = np.array(frame.convert("RGB"))
        if effect == "mask":
            video.highlight_mask(rgb, mask)
        else:
            rgb = stylize(rgb, mask, effect, strength)
        image = Image.fromarray(rgb)
        image.thumbnail((1280, 1280))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=90)
        return Response(
            buffer.getvalue(),
            media_type="image/jpeg",
            headers={"X-Preview-Ready": str(len(job["masks"]))},
        )


@app.post("/api/{sid}/propagate")
def propagate(sid: str, req: Request):
    with LOCK:
        s = session(sid)
        key = tracking_key(s, req)
        token = uuid.uuid4().hex
        out = s["root"] / token
        out.mkdir()
        preview = s.get("live_preview")
        if preview and preview["key"] == key:
            for index in range(len(preview["masks"]), s["count"]):
                advance_preview(s, preview, index)
            masks = preview["masks"]
            video.write_video(s["root"] / "source", s["times"], s["rate"], masks, out)
        else:
            s.pop("live_preview", None)
            del preview
            masks = video.render(
                s["root"] / "frames",
                s["root"] / "source",
                s["times"],
                s["rate"],
                req.keyframes,
                req.version,
                out,
            )
        (out / "prompts.json").write_text(
            json.dumps(
                {
                    "version": req.version,
                    "frames": s["count"],
                    "timestamps": s["times"],
                    "keyframes": {i: p.model_dump() for i, p in req.keyframes.items()},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        windows = video.style_windows(masks, s["w"], s["h"])
        s.update(export=token, export_key=key, tracked_masks=masks, windows=windows)
        return {"export": token}


@app.get("/api/{sid}/tracked/{token}/{index}")
def tracked(sid: str, token: str, index: int):
    with LOCK:
        s = session(sid, index)
        return png(
            Image.fromarray(tracked_mask(s, token, index).astype(np.uint8) * 255)
        )


@app.post("/api/{sid}/video-style/{token}")
def style_video(sid: str, token: str, settings: StyleSettings):
    with LOCK:
        s = session(sid)
        if not s["video"] or s["export"] != token:
            raise HTTPException(409, "请先传播当前视频的掩膜")
        folder = s["root"] / token
        variant = f"{settings.style}-{settings.strength}"
        dest = folder / variant
        if not dest.exists():
            with tempfile.TemporaryDirectory(dir=folder) as temp:
                staging = Path(temp) / "result"
                staging.mkdir()
                video.write_video(
                    s["root"] / "source",
                    s["times"],
                    s["rate"],
                    s["tracked_masks"],
                    staging,
                    settings.style,
                    settings.strength,
                    windows=s["windows"]
                    if settings.style.startswith("neural-")
                    else None,
                )
                staging.rename(dest)
        return {"variant": variant}


@app.get("/api/{sid}/video-style/{token}/{style}/{strength}/{name}")
def download_styled_video(
    sid: str,
    token: str,
    style: Style,
    strength: int,
    name: Literal["styled.mp4", "styled-frames.zip"],
):
    with LOCK:
        s = session(sid)
        if s["export"] != token or not 0 <= strength <= 100:
            raise HTTPException(409, "跟踪结果已更新")
        path = s["root"] / token / f"{style}-{strength}" / name
        if not path.is_file():
            raise HTTPException(404, "请先生成风格视频")
        return FileResponse(
            path, media_type="video/mp4" if name.endswith(".mp4") else "application/zip"
        )


@app.get("/api/{sid}/export/{token}/{name}")
def export(sid: str, token: str, name: str):
    with LOCK:
        s = session(sid)
        types = {
            "preview.mp4": "video/mp4",
            "foreground.zip": "application/zip",
            "prompts.json": "application/json",
        }
        if s["export"] != token or name not in types:
            raise HTTPException(404)
        return FileResponse(s["root"] / token / name, media_type=types[name])


app.mount("/", StaticFiles(directory=WEB, html=True), name="ui")
