"""Manually located wheels, with native-pixel RGB and mask audit crops."""

import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

from sam_app.models import ROOT
from tools.fetch_spoked_wheels import ROOT as DATA

OUT = ROOT / "outputs/local-decoder"
# Image-relative x, y in a 160 x 120 overview; half width in overview x units.
FIELDS = {
    "tr01": (120, 72, 26),
    "tr03": (111, 85, 21),
    "tr05": (80, 72, 27),
    "tr07": (69, 86, 22),
    "tr09": (92, 77, 24),
    "tr12": (120, 54, 26),
    "tr13": (126, 82, 29),
    "tr14": (45, 84, 26),
    "tr15": (107, 76, 27),
    "tr16": (85, 90, 27),
    "tr17": (116, 69, 23),
    "tr18": (108, 89, 28),
    "tr19": (52, 85, 23),
    "tr21": (113, 85, 27),
    "tr22": (57, 87, 12),
    "tr25": (128, 87, 25),
    "tr29": (22, 94, 20),
    "tr30": (61, 61, 51),
    "tr32": (26, 88, 16),
    "tr33": (126, 75, 15),
    "tr35": (38, 94, 19),
    "tr37": (131, 81, 29),
    "tr40": (87, 89, 25),
    "tr41": (112, 67, 38),
    "tr42": (45, 72, 34),
    "tr43": (129, 86, 17),
    "tr44": (35, 62, 14),
    "vd00": (100, 83, 27),
    "vd01": (20, 85, 22),
    "vd02": (115, 70, 25),
    "vd03": (111, 83, 27),
    "vd04": (43, 80, 29),
    "vd05": (116, 80, 27),
    "vd07": (32, 48, 19),
    "vd09": (38, 85, 29),
    "vd10": (76, 83, 32),
    "vd11": (32, 73, 23),
}


def read(key):
    with Image.open(DATA / f"{key}.jpg") as img:
        rgb = np.array(img.convert("RGB"))
    with Image.open(DATA / f"{key}.png") as img:
        gt = np.array(img.convert("L")) >= 128
    return rgb, gt


def crop_rect(key, shape):
    h, w = shape
    x, y, r = FIELDS[key]
    width = min(w, 1050, round(2 * r * w / 160))
    height = min(h, round(width * 1.18))
    x = int(np.clip(round(x * w / 160 - width / 2), 0, w - width))
    y = int(np.clip(round(y * h / 120 - height / 2), 0, h - height))
    return x, y, x + width, y + height


def audit():
    OUT.mkdir(exist_ok=True)
    records = []
    for i, key in enumerate(FIELDS):
        if i % 10 == 0:
            sheet = Image.new("RGB", (4 * 183, 5 * 179), "#e5e6e3")
            pen = ImageDraw.Draw(sheet)
        rgb, gt = read(key)
        x, y, r, b = crop_rect(key, gt.shape)
        mask = gt[y:b, x:r].astype(np.uint8)
        opened = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        )
        fine = mask - opened
        n = 160
        sums = cv2.integral(fine)
        counts = sums[n:, n:] - sums[:-n, n:] - sums[n:, :-n] + sums[:-n, :-n]
        yy, xx = np.unravel_index(counts.argmax(), counts.shape)
        patch = np.s_[y + yy : y + yy + n, x + xx : x + xx + n]
        xx0 = (i % 2) * 366
        yy0 = (i % 10 // 2) * 179
        pen.text((xx0 + 3, yy0 + 2), key, fill="black")
        sheet.paste(Image.fromarray(rgb[patch]), (xx0, yy0 + 17))
        sheet.paste(
            Image.fromarray(gt[patch].astype(np.uint8) * 255), (xx0 + 164, yy0 + 17)
        )
        records.append(
            {
                "id": key,
                "crop": [x, y, r, b],
                "detail": [int(x + xx), int(y + yy), int(x + xx + n), int(y + yy + n)],
            }
        )
        if i % 10 == 9 or i == len(FIELDS) - 1:
            sheet.save(OUT / f"native-audit-{i // 10 + 1}.png")
    (OUT / "screened-fields.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    audit()
