import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/DAVIS"
SPLIT = ROOT / "data/splits.json"


def split_videos(data=DATA, seed=42):
    folder = data / "ImageSets/2017"
    train, val = [
        sorted((folder / f"{name}.txt").read_text().split())
        for name in ("train", "val")
    ]
    if len(set(train)) != 60 or len(set(val)) != 30 or set(train) & set(val):
        raise ValueError("需要 DAVIS 2017 semi-supervised TrainVal 的 60/30 个视频。")
    random.Random(seed).shuffle(train)
    return {
        "seed": seed,
        "train": sorted(train[:50]),
        "dev": sorted(train[50:]),
        "eval": val,
    }


def samples(data, names, per_video):
    result = []
    for name in names:
        labels = sorted((data / "Annotations/480p" / name).glob("*.png"))
        if not labels:
            raise ValueError(f"缺少标注 {name}")
        ids = np.linspace(0, len(labels) - 1, min(per_video, len(labels)), dtype=int)
        for index in ids:
            lbl = labels[index]
            img = data / "JPEGImages/480p" / name / (lbl.stem + ".jpg")
            if not img.is_file():
                raise FileNotFoundError(img)
            result.append((img, lbl))
    return result


def read_target(pair, rng):
    img_path, label_path = pair
    with Image.open(img_path) as im:
        image = np.array(im.convert("RGB"))
    with Image.open(label_path) as label:
        ids = np.array(label)
    if ids.ndim != 2 or ids.shape != image.shape[:2]:
        raise ValueError(f"标注格式或尺寸错误: {label_path}")
    objects = np.setdiff1d(np.unique(ids), [0, 255])
    if not len(objects):
        return None
    obj = int(rng.choice(objects))
    return image, ids == obj, ids != 255, obj


def prompt(mask, rng, mode, jitter=0.2):
    ys, xs = np.where(mask)
    if mode == "point":
        i = rng.integers(len(xs))
        return [[int(xs[i]), int(ys[i])]], [1], None
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    w, h = mask.shape[1], mask.shape[0]
    delta = rng.uniform(-jitter, jitter, 4) * np.array(
        [x1 - x0, y1 - y0, x1 - x0, y1 - y0]
    )
    box = np.clip(np.array([x0, y0, x1, y1]) + delta, 0, [w - 1, h - 1, w - 1, h - 1])
    return [], [], box.tolist()


def scores(mask, target, valid):
    a, b = mask & valid, target & valid
    inter = int((a & b).sum())
    return inter / max(1, (a | b).sum()), 2 * inter / max(1, a.sum() + b.sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    split = split_videos(args.data)
    for key in ("train", "dev", "eval"):
        print(
            key,
            len(split[key]),
            "videos,",
            len(samples(args.data, split[key], 10)),
            "sampled frames",
        )
    SPLIT.write_text(json.dumps(split, indent=2), encoding="utf-8")
