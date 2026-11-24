"""Evaluate first-frame referring expressions with the original text detector."""

import argparse
import csv
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from sam_app import models
from training.data import DATA, SPLIT, scores


def box_iou(a, b):
    left, top = np.maximum(a[:2], b[:2])
    right, bottom = np.minimum(a[2:], b[2:])
    overlap = max(0, right - left) * max(0, bottom - top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(overlap / max(1, area_a + area_b - overlap))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "eval"], default="dev")
    p.add_argument("--annotator", type=int, choices=[1, 2], default=1)
    p.add_argument("--limit", type=int)
    p.add_argument("--version", choices=["original", "tuned"], default="original")
    p.add_argument("--out", type=Path, default=models.ROOT / "outputs/text-evaluation")
    args = p.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    selected = set(json.loads(SPLIT.read_text())[args.split])
    with ZipFile(models.ROOT / "data/davis_text_annotations.zip") as z:
        content = z.read(
            f"davis_text_annotations/Davis17_annot{args.annotator}.txt"
        ).decode("utf-8-sig")
    queries = []
    for line in content.splitlines():
        if not line.strip():
            continue
        seq, obj, phrase = line.split(maxsplit=2)
        if seq in selected:
            queries.append((seq, int(obj), phrase.strip('" ')))
    if args.limit:
        queries = queries[: args.limit]
    engine = models.predictor(args.version)
    rows = []
    for seq, obj, phrase in queries:
        first = min((DATA / "Annotations/480p" / seq).glob("*.png"))
        with Image.open(first) as im:
            ids = np.array(im)
        with Image.open(DATA / "JPEGImages/480p" / seq / (first.stem + ".jpg")) as im:
            rgb = np.array(im.convert("RGB"))
        target = ids == obj
        if not target.any():
            raise ValueError(f"目标缺少首帧标注 {seq} {obj}")
        candidates = models.candidates(rgb, phrase)
        overlap = mask_iou = 0.0
        if candidates:
            ys, xs = np.where(target)
            gt_box = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
            box = candidates[0]["box"]
            overlap = box_iou(box, gt_box)
            engine.set_image(rgb)
            mask = models.mask_from_prompt(engine, [], [], box)
            mask_iou, _ = scores(mask, target, ids != 255)
        rows.append([seq, obj, phrase, len(candidates), overlap, mask_iou])
    stem = f"{args.split}-{args.version}-annot{args.annotator}"
    with (args.out / (stem + ".csv")).open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(
            ["video", "object", "description", "candidates", "top_box_iou", "mask_iou"]
        )
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "count": len(rows),
                "top1_hit_rate": np.mean([r[4] >= 0.5 for r in rows]),
                "mask_iou": np.mean([r[5] for r in rows]),
            }
        )
    )


if __name__ == "__main__":
    main()
