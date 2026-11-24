"""Render registered wheel cutouts and a paired audit sheet for every sampled field."""

import csv
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.plot_davis_resolution import BOLD, FONT
from training.audit_local_sam import BRANCHES, DETAILS, FIELDS, OUT

PAPER, INK, MUTED, ACCENT = "#f6f6f2", "#182729", "#485b5b", "#ecab2c"
HEADINGS = (
    "全分辨率原图",
    "DAVIS 官方标注",
    "整图 SAM",
    "480p 局部 SAM",
    "全分辨率局部 SAM",
)
NAMES = {"point": "单点", "box": "紧框"}


def label(d, x, y, s, size=18, bold=False, color=INK):
    d.text((x, y), s, font=ImageFont.truetype(BOLD if bold else FONT, size), fill=color)


def binary(mask, valid):
    return Image.fromarray(
        np.where(valid, mask.astype(np.uint8) * 255, 127).astype(np.uint8)
    ).convert("RGB")


def wheel(out, data, case, records):
    rgb = Image.fromarray(data["rgb"])
    truth = binary(data["truth"], data["valid"])
    cw, gap, left = 268, 12, 24
    width = 2 * left + 5 * cw + 4 * gap
    canvas = Image.new("RGB", (width, 1028), PAPER)
    d = ImageDraw.Draw(canvas)
    label(d, left, 10, "原来的自行车车轮：同一次选中，三种推理", 33, True)
    label(
        d,
        left,
        61,
        "bike-packing / 00000  ·  实例 1  ·  同一 536 × 759 原生像素区域  ·  白色为前景，黑色为背景",
        19,
        color=MUTED,
    )
    for j, title in enumerate(HEADINGS):
        label(d, left + j * (cw + gap), 106, title, 22, True)
    marked = rgb.copy()
    md = ImageDraw.Draw(marked)
    for letter, (x, y, r, b, _) in DETAILS.items():
        md.rectangle((x - 2, y - 2, r + 1, b + 1), outline=ACCENT, width=3)
        label(md, x, y - 37, letter, 30, True, ACCENT)
    for i, (mode, title) in enumerate(NAMES.items()):
        y = 151 + i * 417
        panels = [marked, truth] + [
            binary(data[f"{mode}_{k}"], data["valid"]) for k in BRANCHES
        ]
        for j, item in enumerate(panels):
            x = left + j * (cw + gap)
            tag = title if j == 0 else ""
            if j >= 2:
                rec = next(
                    r
                    for r in records
                    if r["mode"] == mode and r["branch"] == BRANCHES[j - 2]
                )
                tag = f"轮廓 F@2   {float(rec['f2']):.3f}"
            label(d, x, y, tag, 18, False, MUTED)
            canvas.paste(
                item.resize(
                    (268, 380),
                    Image.Resampling.LANCZOS if j == 0 else Image.Resampling.NEAREST,
                ),
                (x, y + 29),
            )
    label(
        d,
        left,
        990,
        "A 与 B 为固定核对区。两个局部分支共享该行的全图粗掩膜。两组都沿用原来的操作。",
        17,
        color=MUTED,
    )
    canvas.save(out / "wheel-comparison.png")

    # Pixel magnification is identical for all panels in a row, including RGB.
    scale = 3
    cw = 369
    width = 2 * left + 5 * cw + 4 * gap
    canvas = Image.new("RGB", (width, 1040), PAPER)
    d = ImageDraw.Draw(canvas)
    label(d, left, 13, "放大 A、B 两处：哪些辐条留下，哪些空隙被填", 37, True)
    label(
        d,
        left,
        71,
        "各列相同视野，原生像素放大 3 倍。原图左侧短横标出核对线高度；白色为标注或预测的前景。",
        23,
        color=MUTED,
    )
    for j, title in enumerate(HEADINGS):
        label(d, left + j * (cw + gap), 125, title, 25, True)
    top = 178
    for mode in ("box", "point"):
        for letter, (x, y, r, b, cut) in DETAILS.items():
            block = case["transects"][letter]
            panels = [rgb, truth] + [
                binary(data[f"{mode}_{k}"], data["valid"]) for k in BRANCHES
            ]
            for j, item in enumerate(panels):
                xx = left + j * (cw + gap)
                yy = top + 37
                note = NAMES[mode] + "  ·  " + letter if j == 0 else "官方二值掩膜"
                if j >= 2:
                    score = block["scores"][f"{mode}_{BRANCHES[j - 2]}"]
                    note = f"辐条覆盖 {score['spoke_tp'] / block['fg_pixels']:.0%}    空隙误填 {score['gap_fp'] / block['bg_pixels']:.0%}"
                label(d, xx, top, note, 20, bold=j == 0)
                img = item.crop((x, y, r, b)).resize(
                    ((r - x) * scale, (b - y) * scale), Image.Resampling.NEAREST
                )
                canvas.paste(img, (xx, yy))
                if j == 0:
                    cy = yy + scale * (cut - y) + 1
                    d.line((xx - 12, cy, xx - 2, cy), fill="#945300", width=3)
            top += 37 + (b - y) * scale + 18
    label(
        d,
        left,
        top + 3,
        "百分比在各自矩形内计算。辐条覆盖以标注前景为分母，空隙误填以标注背景为分母。两项都需查看。",
        21,
        color=MUTED,
    )
    canvas.crop((0, 0, width, top + 53)).save(out / "wheel-spokes.png")


def sample_sheets():
    for page in range(2):
        canvas = Image.new("RGB", (3 * 391, 5 * 226 + 46), PAPER)
        d = ImageDraw.Draw(canvas)
        label(
            d,
            8,
            3,
            "全分辨率标注核对    每格左为原图，右为同一实例标注。白色为前景，灰色为忽略。",
            18,
        )
        for j, (video, items) in enumerate(
            list(FIELDS.items())[5 * page : 5 * page + 5]
        ):
            for i, (frame, *_) in enumerate(items):
                x, y = i * 391, 38 + j * 226
                label(d, x + 2, y, f"{video} / {frame}", 16)
                for k, suffix in enumerate(("rgb", "label")):
                    with Image.open(
                        OUT / "fields" / f"{video}-{frame}-{suffix}.png"
                    ) as img:
                        canvas.paste(img, (x + k * 193, y + 26))
        canvas.save(OUT / f"audit-sheet-{page + 1}.png")


if __name__ == "__main__":
    case = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))["cases"][0]
    with (OUT / "metrics.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["case"] == case["key"]]
    with np.load(OUT / "wheel.npz") as data:
        wheel(OUT, data, case, rows)
    sample_sheets()
