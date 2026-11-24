"""Registered wheel details, rigid-part checks and foreground change map."""

import json

import numpy as np
from PIL import Image, ImageDraw

from tools.plot_wheel_crop_audit import ACCENT, MUTED, PAPER, binary, label
from training.audit_local_sam import DETAILS
from training.evaluate_local_wheels import PARTS
from training.wheel_samples import OUT

MODES = {"box": "紧框", "point": "单点"}


def spokes(z, res):
    canvas = Image.new("RGB", (1600, 1055), PAPER)
    d = ImageDraw.Draw(canvas)
    label(d, 24, 13, "车轮辐条  |  局部解码器微调前后", 37, True)
    label(
        d,
        24,
        70,
        "DAVIS  bike-packing / 00000  ·  全分辨率原图  ·  相同用户提示、局部视野与粗掩膜  ·  每个原生像素放大 3 倍",
        21,
        color=MUTED,
    )
    for j, name in enumerate(("原图", "官方标注", "训练前", "训练后")):
        label(d, 24 + j * 395, 126, name, 25, True)
    y = 177
    for mode, name in MODES.items():
        for code, (x, b, r, t, cut) in DETAILS.items():
            before = f"{mode}_before"
            after = f"{mode}_{res['chosen']}"
            images = [
                Image.fromarray(z["rgb"]),
                binary(z["truth"], z["valid"]),
                binary(z[before], z["valid"]),
                binary(z[after], z["valid"]),
            ]
            for j, im in enumerate(images):
                xx = 24 + j * 395
                cap = f"{name}  ·  {code}" if j == 0 else "白色为辐条前景"
                if j > 1:
                    s = res["scores"][[before, after][j - 2]][code]
                    cap = f"覆盖 {s['recall']:.1%}     空隙误填 {s['gap_fp']:.1%}"
                label(d, xx, y, cap, 20, bold=j == 0)
                im = im.crop((x, b, r, t)).resize(
                    ((r - x) * 3, (t - b) * 3), Image.Resampling.NEAREST
                )
                canvas.paste(im, (xx, y + 36))
                yy = y + 36 + (cut - b) * 3 + 1
                d.line((xx - 11, yy, xx - 2, yy), fill="#965300", width=3)
            y += 36 + (t - b) * 3 + 21
    cut = res["transects"]["B"]["scores"]
    missed = [
        str(i + 1)
        for i, (box, point) in enumerate(
            zip(
                cut[f"box_{res['chosen']}"]["fg_run_hit_pixels"],
                cut[f"point_{res['chosen']}"]["fg_run_hit_pixels"],
                strict=True,
            )
        )
        if box == point == 0
    ]
    desc = "、".join(missed) if missed else "无"
    label(
        d,
        24,
        y + 6,
        f"短横线标出同一水平核对线。B 区从左到右编号，两种提示训练后仍完全漏检的前景段为 {desc}。",
        20,
        color=MUTED,
    )
    canvas.crop((0, 0, canvas.width, y + 50)).save(OUT / "spokes-before-after.png")


def parts(z, res):
    canvas = Image.new("RGB", (1460, 1075), PAPER)
    d = ImageDraw.Draw(canvas)
    label(d, 24, 12, "轮胎与车架  |  同一车轮区域的保留与误删", 35, True)
    label(
        d,
        24,
        64,
        "蓝色为训练后找回的标注前景，橙色为训练后新增误删的标注前景，灰色为前后都漏掉的前景。",
        20,
        color=MUTED,
    )
    for j, title in enumerate(
        ("核对位置", "官方标注", "训练前", "训练后", "前景像素变化")
    ):
        label(d, 24 + j * 284, 118, title, 23, True)
    original = Image.fromarray(z["rgb"])
    noted = original.copy()
    nd = ImageDraw.Draw(noted)
    for k, poly in PARTS.items():
        nd.line(poly + [poly[0]], fill=ACCENT, width=4)
        tx, ty = (86, 26) if k == "tire" else (428, 318)
        label(nd, tx, ty, "轮胎" if k == "tire" else "车架", 25, True, ACCENT)
    for code, (x, y, r, b, _) in DETAILS.items():
        nd.rectangle((x, y, r, b), outline="white", width=2)
        label(nd, x, y - 23, code, 20, True, "white")
    y = 160
    for mode, name in MODES.items():
        k = f"{mode}_{res['chosen']}"
        prev = z[f"{mode}_before"]
        after = z[k]
        gt = z["truth"] & z["valid"]
        changes = np.zeros((*gt.shape, 3), np.uint8) + 25
        changes[gt & prev & after] = [219, 223, 219]
        changes[gt & ~prev & ~after] = [108, 108, 108]
        changes[gt & ~prev & after] = [60, 161, 232]
        changes[gt & prev & ~after] = [255, 143, 50]
        for j, img in enumerate(
            (
                noted,
                binary(z["truth"], z["valid"]),
                binary(prev, z["valid"]),
                binary(after, z["valid"]),
                Image.fromarray(changes),
            )
        ):
            xx = 24 + j * 284
            cap = name if j == 0 else ""
            if j == 2 or j == 3:
                stat = res["scores"][[f"{mode}_before", k][j - 2]]["wheel"]
                cap = f"IoU {stat['iou']:.3f}    F@2 {stat['f2']:.3f}"
            label(d, xx, y, cap, 18, color=MUTED)
            canvas.paste(
                img.resize(
                    (265, 375),
                    Image.Resampling.LANCZOS if j == 0 else Image.Resampling.NEAREST,
                ),
                (xx, y + 31),
            )
        tire, frame = res["scores"][k]["tire"], res["scores"][k]["frame"]
        label(
            d,
            24,
            y + 410,
            f"{name}核对区    轮胎找回 {tire['recovered']} 像素、新增误删 {tire['newly_lost']} 像素    |    车架找回 {frame['recovered']} 像素、新增误删 {frame['newly_lost']} 像素",
            18,
            color=MUTED,
        )
        y += 451
    canvas.save(OUT / "tire-frame-before-after.png")


if __name__ == "__main__":
    scores = json.loads((OUT / "wheel-scores.json").read_text(encoding="utf-8"))
    with np.load(OUT / "wheel-predictions.npz") as z:
        spokes(z, scores)
        parts(z, scores)
