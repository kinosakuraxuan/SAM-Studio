"""Show the matched high-resolution bicycle crop and the paired predictions."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from training.compare_davis_resolution import OUTPUT

BACKGROUND, TEXT, SECONDARY = "#f5f7f7", "#16272b", "#365a60"
GOLD, MINT, BETTER, WORSE = (
    (255, 221, 80),
    (55, 241, 220),
    (40, 170, 255),
    (255, 153, 58),
)
FONT = "C:/Windows/Fonts/msyh.ttc"
BOLD = "C:/Windows/Fonts/msyhbd.ttc"


def paint(rgb, mask, valid, color):
    region = mask & valid
    inside = cv2.erode(region.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    result = (rgb * 0.76).astype(np.uint8)
    result[region & ~inside] = color
    return result


def paste(canvas, draw, im, x, y, width, height, size):
    draw.rounded_rectangle((x, y, x + width, y + height), radius=7, fill="#e3ecee")
    pic = Image.fromarray(im).resize(size, Image.Resampling.NEAREST)
    canvas.paste(pic, (x + (width - pic.width) // 2, y + (height - pic.height) // 2))


def fonts():
    return [
        ImageFont.truetype(path, s) for path, s in [(BOLD, 40), (BOLD, 25), (FONT, 19)]
    ]


def render_comparison(output, data, meta):
    title, header, body = fonts()
    ys, xs = data["truth"].shape
    w, h, gutter, margin, top = max(330, xs), max(465, ys), 14, 24, 110
    row = h + 85
    canvas = Image.new(
        "RGB", (2 * margin + 5 * w + 4 * gutter, top + 2 * row + 63), BACKGROUND
    )
    draw = ImageDraw.Draw(canvas)
    zoom = min(w / xs, h / ys)
    size = round(xs * zoom), round(ys * zoom)
    draw.text((margin, 14), "同一自行车车轮的分辨率对照", font=title, fill=TEXT)
    full, low = meta["full_size"], meta["low_size"]
    draw.text(
        (margin, 74),
        f"DAVIS 2017 · bike-packing/00000   |   全分辨率 {full[0]} × {full[1]}   |   低清输入 {low[0]} × {low[1]}   |   同一全分辨率标注评分",
        font=body,
        fill=SECONDARY,
    )
    rgb, valid, gt = data["rgb"], data["valid"], data["truth"]
    for i, (prompt, name) in enumerate((("point", "单点"), ("box", "紧框"))):
        y = top + i * row
        panels = [
            (
                f"{name} · 全分辨率标注",
                "黄线为官方实例轮廓",
                paint(rgb, gt, valid, GOLD),
            ),
            ("480p 输入细节", "映射到相同的像素网格", data["low"]),
        ]
        for key, caption in (
            ("coarse", "共享 SAM 粗掩膜"),
            ("480p", "480p 训练版"),
            ("full", "全分辨率训练版"),
        ):
            metrics = meta["metrics"][prompt][key]
            share = metrics["thin_preserved"] / max(1, metrics["thin_pixels"])
            detail = f"F@2 {metrics['local_f2']:.4f}  ·  细结构召回 {share:.1%}"
            panels.append(
                (caption, detail, paint(rgb, data[f"{prompt}_{key}"], valid, MINT))
            )
        for j, (caption, detail, im) in enumerate(panels):
            x = margin + j * (w + gutter)
            draw.text((x, y + 1), caption, font=header, fill=TEXT)
            draw.text((x, y + 36), detail, font=body, fill=SECONDARY)
            paste(canvas, draw, im, x, y + 67, w, h, size)
        draw.line(
            (margin, y + row - 7, canvas.width - margin, y + row - 7), fill="#cbdadc"
        )
    draw.text(
        (margin, top + 2 * row + 13),
        "所有轮廓使用同样的区域与缩放。F@2 按全分辨率原图像素计算。细结构由全分辨率标注上的 5×5 开运算近似。",
        font=body,
        fill=SECONDARY,
    )
    canvas.save(output / "wheel-resolution-comparison.png")


def render_changes(output, data):
    title, header, body = fonts()
    w, h, margin = 375, 485, 25
    canvas = Image.new("RGB", (4 * w + 5 * margin, h + 177), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 15), "全分辨率版在哪些像素改变了判定", font=title, fill=TEXT)
    truth, valid, thin = data["truth"], data["valid"], data["thin"]
    height, width = truth.shape
    cw, ch = min(178, width), min(222, height)
    integral = cv2.integral(thin.astype(np.uint8))
    counts = (
        integral[ch:, cw:]
        - integral[:-ch, cw:]
        - integral[ch:, :-cw]
        + integral[:-ch, :-cw]
    )
    y, x = np.unravel_index(counts.argmax(), counts.shape)
    window = np.s_[y : y + ch, x : x + cw]
    scale = max(1, min(w // cw, h // ch))
    size = cw * scale, ch * scale
    draw.text(
        (margin, 73),
        "取标注细结构最密集处。蓝色是全分辨率版改对或新保留，橙色是改错或新漏掉，青色是两版都覆盖的标注细结构。",
        font=body,
        fill=SECONDARY,
    )
    for i, (mode, mark) in enumerate((("point", "单点"), ("box", "紧框"))):
        low, full = data[f"{mode}_480p"], data[f"{mode}_full"]
        correct = valid & (low != truth) & (full == truth)
        wrong = valid & (low == truth) & (full != truth)
        diff = (data["rgb"] * 0.76).astype(np.uint8)
        diff[correct], diff[wrong] = BETTER, WORSE
        targeted = (data["rgb"] * 0.72).astype(np.uint8)
        targeted[thin & low & full] = MINT
        targeted[thin & ~low & full] = BETTER
        targeted[thin & low & ~full] = WORSE
        for j, (caption, sub, img) in enumerate(
            [
                (
                    f"{mark} · 全部变化",
                    f"改对 {correct[window].sum()} px · 改错 {wrong[window].sum()} px",
                    diff,
                ),
                (
                    f"{mark} · 标注细结构",
                    f"新保留 {(thin & ~low & full)[window].sum()} px · 新漏掉 {(thin & low & ~full)[window].sum()} px",
                    targeted,
                ),
            ]
        ):
            x = margin + (i * 2 + j) * (w + margin)
            draw.text((x, 108), caption, font=header, fill=TEXT)
            draw.text((x, 140), sub, font=body, fill=SECONDARY)
            paste(canvas, draw, img[window], x, 171, w, h, size)
    canvas.save(output / "wheel-resolution-changes.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUTPUT)
    args = p.parse_args()
    output = args.out / "evaluation"
    meta = json.loads((output / "wheel.json").read_text(encoding="utf-8"))
    with np.load(output / "wheel.npz") as data:
        render_comparison(output, data, meta)
        render_changes(output, data)


if __name__ == "__main__":
    main()
