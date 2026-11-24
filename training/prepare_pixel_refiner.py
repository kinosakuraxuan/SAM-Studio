"""Cache frozen SAM logits and aligned DAVIS patches for both ablations."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sam_app import models
from sam_app.pixel_refiner import refinement_bounds
from training.data import DATA, SPLIT, prompt, read_target, samples, scores

DEFAULT_OUT = models.ROOT / "outputs/pixel-refiner"
SEED = 42
PATCH = 128
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def thin_pixels(target, valid):
    opened = cv2.morphologyEx((target & valid).astype(np.uint8), cv2.MORPH_OPEN, KERNEL)
    return target & valid & ~opened.astype(bool)


def training_patches(rgb, coarse, target, valid, bounds, rng):
    for window, weights in training_windows(target, valid, bounds, rng):
        yield rgb[window], coarse[window], target[window], weights


def training_windows(target, valid, bounds, rng):
    x0, y0, x1, y1 = bounds
    region = np.s_[y0:y1, x0:x1]
    boundary = cv2.morphologyEx(target.astype(np.uint8), cv2.MORPH_GRADIENT, KERNEL)
    thin = thin_pixels(target, valid)
    weights = (1 + 3 * boundary + 3 * thin) * valid
    for pool in (thin, boundary > 0, valid):
        ys, xs = np.where((pool & valid)[region])
        if len(xs):
            chosen = rng.integers(len(xs))
            x, y = xs[chosen] + x0, ys[chosen] + y0
        else:
            x, y = (x0 + x1) // 2, (y0 + y1) // 2
        h, w = target.shape
        left = max(0, min(x - PATCH // 2, w - PATCH))
        top = max(0, min(y - PATCH // 2, h - PATCH))
        window = np.s_[top : top + PATCH, left : left + PATCH]
        yield window, weights[window]


def prepare_split(engine, split, sequences, per_video, cache, limit):
    pairs = samples(DATA, sequences, per_video)
    if limit:
        pairs = pairs[:limit]
    rng = np.random.default_rng(SEED if split == "train" else SEED + 1)
    entries, patch_count, rejected, thin_count = [], 0, 0, 0
    directory = cache / split
    directory.mkdir(exist_ok=True)
    for i, pair in enumerate(pairs):
        item = read_target(pair, rng)
        if item is None:
            continue
        rgb, target, valid, instance = item
        engine.set_image(rgb)
        plans = {
            "point": prompt(target, rng, "point"),
            "box": prompt(
                target, rng, "box", rng.choice([0, 0.1, 0.2]) if split == "train" else 0
            ),
        }
        path = directory / f"{i:04d}.npz"
        if split == "train":
            patches = []
            for points in plans.values():
                coarse = models.prompt_prediction(engine, *points)
                bounds = refinement_bounds(coarse, *points)
                # Teach boundary recovery on prompts that already select the target.
                if bounds is None or scores(coarse > 0, target, valid)[0] < 0.45:
                    rejected += 1
                    continue
                patches.extend(
                    training_patches(rgb, coarse, target, valid, bounds, rng)
                )
            if not patches:
                continue
            pixels, logits, truth, weights = zip(*patches)
            np.savez_compressed(
                path,
                rgb=np.stack(pixels),
                coarse=np.stack(logits).astype(np.float16),
                truth=np.stack(truth),
                weights=np.stack(weights).astype(np.uint8),
            )
            patch_count += len(patches)
        else:
            logits = {
                name: models.prompt_prediction(engine, *p) for name, p in plans.items()
            }
            np.savez_compressed(path, **logits)
        thin_count += int(thin_pixels(target, valid).sum())
        entries.append(
            {
                "image": pair[0].relative_to(DATA).as_posix(),
                "mask": pair[1].relative_to(DATA).as_posix(),
                "object": instance,
                "prompts": plans,
                "cache": path.relative_to(cache).as_posix(),
            }
        )
        if (i + 1) % 50 == 0:
            print(f"{split}: {i + 1}/{len(pairs)} frames processed", flush=True)
    return {
        "frames": entries,
        "patches": patch_count,
        "skipped_train_prompts": rejected,
        "thin_foreground_pixels": thin_count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, help="每个划分的冒烟测试帧数上限")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("帧数上限必须为正数。")
    cache = args.out / "cache"
    target = cache / "manifest.json"
    if target.exists():
        parser.error("已有缓存，请指定其他输出目录。")
    cache.mkdir(parents=True, exist_ok=True)
    splits = json.loads(SPLIT.read_text())
    decoder_digest = models.tuned_decoder_digest()
    engine = models.predictor("tuned")
    summary = {
        "seed": SEED,
        "patch_size": PATCH,
        "base_decoder_sha256": decoder_digest,
        "train": prepare_split(engine, "train", splits["train"], 10, cache, args.limit),
        "dev": prepare_split(engine, "dev", splits["dev"], 3, cache, args.limit),
    }
    if models.tuned_decoder_digest() != decoder_digest:
        raise RuntimeError("生成缓存时微调权重发生变化，请重试。")
    if not summary["train"]["patches"] or not summary["dev"]["frames"]:
        raise ValueError("训练裁剪或开发集为空。")
    summary["sampling"] = ["thin foreground", "boundary", "ordinary"]
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        {
            name: {
                "frames": len(summary[name]["frames"]),
                "patches": summary[name]["patches"],
                "skipped_train_prompts": summary[name]["skipped_train_prompts"],
            }
            for name in ("train", "dev")
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
