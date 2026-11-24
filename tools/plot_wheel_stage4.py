"""Visualize identical native-pixel wheel fields for the two trained models."""

import json

import numpy as np
from PIL import Image, ImageDraw

from tools.plot_wheel_crop_audit import MUTED, PAPER, binary, label
from training.audit_local_sam import DETAILS
from training.local_wheel_stage4 import OUT

MODES = {"box": "紧框", "point": "单点"}
MARK = "#9e4b0a"


def panels(z, mode):
    return [
        Image.fromarray(z["rgb"]),
        binary(z["truth"], z["valid"]),
        binary(z[f"{mode}_decoder"], z["valid"]),
        binary(z[f"{mode}_joint"], z["valid"]),
    ]


def details(z, result):
    width = 1600
    canvas = Image.new("RGB", (width, 1140), PAPER)
    d = ImageDraw.Draw(canvas)
    label(d, 24, 13, "车轮 A、B 局部与第 3 段辐条", 36, True)
    label(
        d,
        24,
        65,
        "同一原图与标注，原生像素放大 3 倍。B 图上缘三角对准第 3 段，左侧短横对准核对行，两者交点为核对位置。",
        21,
        color=MUTED,
    )
    for j, title in enumerate(
        ("原图", "DAVIS 官方标注", "仅训练解码器", "末阶段＋解码器")
    ):
        label(d, 24 + j * 395, 120, title, 25, True)
    y = 168
    for mode, title in MODES.items():
        for code, (x, b, r, t, cut) in DETAILS.items():
            for j, img in enumerate(panels(z, mode)):
                xx = 24 + j * 395
                cap = "白色为辐条前景" if j == 1 else f"{title} · {code}"
                if j >= 2:
                    arm = ("decoder", "joint")[j - 2]
                    s = result["scores"][f"{mode}_{arm}"][code]
                    cap = f"覆盖 {s['recall']:.1%}     空隙误填 {s['gap_fp']:.1%}"
                label(d, xx, y, cap, 19, bold=j == 0)
                yy = y + 44
                pic = img.crop((x, b, r, t)).resize(
                    ((r - x) * 3, (t - b) * 3), Image.Resampling.NEAREST
                )
                canvas.paste(pic, (xx, yy))
                if code == "B":
                    left, right = result["transects"]["B"]["foreground_runs"][2]
                    marker = xx + int(((left + right) / 2 - x) * 3)
                    d.polygon(
                        [
                            (marker - 5, yy - 10),
                            (marker + 5, yy - 10),
                            (marker, yy - 2),
                        ],
                        fill=MARK,
                    )
                    cy = yy + (cut - b) * 3 + 1
                    d.line((xx - 11, cy, xx - 2, cy), fill=MARK, width=3)
            y += 44 + (t - b) * 3 + 20
    counts = result["transects"]["B"]["scores"]
    vals = [
        f"{title} {counts[mode + '_decoder']['fg_run_hit_pixels'][2]}/3 → {counts[mode + '_joint']['fg_run_hit_pixels'][2]}/3"
        for mode, title in MODES.items()
    ]
    label(
        d,
        24,
        y + 5,
        "第 3 段在核对行上的命中像素     " + "      |      ".join(vals),
        21,
        True,
    )
    canvas.crop((0, 0, width, y + 56)).save(OUT / "spokes.png")


def whole(z, result):
    canvas = Image.new("RGB", (1460, 1070), PAPER)
    d = ImageDraw.Draw(canvas)
    label(d, 24, 12, "整轮背景的误填位置", 36, True)
    label(
        d,
        24,
        66,
        "误填变化图仅统计官方背景。橙色为新误填，蓝色为已纠正，灰色为两版共同误填。",
        21,
        color=MUTED,
    )
    for j, title in enumerate(
        ("原图与核对位置", "官方标注", "仅训练解码器", "末阶段＋解码器", "背景误填变化")
    ):
        label(d, 24 + 284 * j, 116, title, 21, True)
    y = 162
    for mode, title in MODES.items():
        images = panels(z, mode)
        d0 = ImageDraw.Draw(images[0])
        for code, (x, b, r, t, _) in DETAILS.items():
            d0.rectangle((x, b, r, t), outline="#facc50", width=3)
            label(d0, x, b - 30, code, 24, True, "#facc50")
        prev = z[mode + "_decoder"]
        cur = z[mode + "_joint"]
        bg = z["valid"] & ~z["truth"]
        change = np.full((*bg.shape, 3), 24, np.uint8)
        change[bg & prev & cur] = [143, 148, 150]
        change[bg & ~prev & cur] = [255, 143, 50]
        change[bg & prev & ~cur] = [60, 161, 232]
        images.append(Image.fromarray(change))
        for j, img in enumerate(images):
            xx = 24 + j * 284
            cap = title if j == 0 else ""
            if j in (2, 3):
                s = result["scores"][mode + ("_decoder" if j == 2 else "_joint")][
                    "wheel"
                ]
                cap = f"误填 {s['gap_fp']:.1%}    IoU {s['iou']:.3f}"
            label(d, xx, y, cap, 18, color=MUTED)
            canvas.paste(
                img.resize(
                    (265, 375),
                    Image.Resampling.LANCZOS if j == 0 else Image.Resampling.NEAREST,
                ),
                (xx, y + 31),
            )
        s = result["scores"][mode + "_joint"]["wheel"]
        label(
            d,
            24,
            y + 409,
            f"{title}  ·  新误填 {s['new_background_errors']:,} 个背景像素，纠正 {s['corrected_background']:,} 个。",
            18,
            color=MUTED,
        )
        y += 448
    canvas.save(OUT / "wheel-background.png")


if __name__ == "__main__":
    res = json.loads((OUT / "wheel.json").read_text(encoding="utf-8"))
    with np.load(OUT / "wheel.npz") as z:
        details(z, res)
        whole(z, res)
