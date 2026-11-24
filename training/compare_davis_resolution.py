"""Paired DAVIS experiment isolating the information retained in the RGB input."""

import argparse
import copy
import csv
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from statistics import mean
from time import perf_counter
from zipfile import ZipFile

import cv2
import numpy as np
import torch
from PIL import Image

from sam_app import models
from sam_app.pixel_refiner import PixelRefiner, refine_pixels, refinement_bounds
from training.data import DATA, scores
from training.evaluate_pixel_refiner import load_refiner
from training.metrics import db_eval_boundary
from training.prepare_pixel_refiner import PATCH, SEED, thin_pixels, training_windows
from training.train_pixel_refiner import patch_loss

ARCHIVE = models.ROOT / "data/DAVIS-2017-trainval-Full-Resolution.zip"
OUTPUT = models.ROOT / "outputs/davis-resolution"
SOURCE = models.ROOT / "outputs/pixel-refiner/cache/manifest.json"
PROMPTS = models.ROOT / "outputs/refinement-evaluation/prompts.json"
CONDITIONS = ("full", "480p")
EPOCHS = 5
BATCH = 8
WHEEL_ROI = (264, 188, 391, 368)


def frame_from_archive(archive, record, edition):
    paths = {}
    for role in ("image", "mask"):
        parts = Path(record[role]).parts
        paths[role] = "/".join(("DAVIS", parts[0], edition, *parts[2:]))
    with Image.open(BytesIO(archive.read(paths["image"]))) as img:
        rgb = np.array(img.convert("RGB"))
    with Image.open(BytesIO(archive.read(paths["mask"]))) as img:
        labels = np.array(img)
    with Image.open(DATA / record["mask"]) as img:
        low_labels = np.array(img)
    full_shape, size = labels.shape, low_labels.shape
    if full_shape != rgb.shape[:2] or labels.ndim != 2:
        raise ValueError(f"全分辨率图像和标签对不齐 {paths}")
    target, valid = labels == record["object"], labels != 255
    if not target.any() or not (low_labels == record["object"]).any():
        raise ValueError(f"配对帧缺少目标实例 {paths}")
    h, w = full_shape
    lh, lw = size
    if abs(w / h - lw / lh) / (w / h) > 0.015:
        raise ValueError(f"两个数据规格的视野比例不一致 {paths}")
    aligned = (
        cv2.resize(
            (low_labels == record["object"]).astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST_EXACT,
        )
        > 0
    )
    overlap = scores(aligned, target, valid)[0]
    if overlap < 0.45:
        raise ValueError(
            f"目标实例的全分辨率与 480p 标注无法可靠配对 {paths}, IoU={overlap:.3f}"
        )
    reduced = cv2.resize(rgb, (lw, lh), interpolation=cv2.INTER_AREA)
    low_rgb = cv2.resize(reduced, (w, h), interpolation=cv2.INTER_LINEAR)
    return rgb, reduced, low_rgb, target, valid, overlap


def scaled_prompt(prompt, low_shape, high_shape):
    h, w = high_shape
    lh, lw = low_shape
    sx, sy = w / lw, h / lh
    points, labels, box = prompt
    points = [
        [
            float(np.clip((x + 0.5) * sx - 0.5, 0, w - 1)),
            float(np.clip((y + 0.5) * sy - 0.5, 0, h - 1)),
        ]
        for x, y in points
    ]
    if box is not None:
        box = [value * scale for value, scale in zip(box, (sx, sy, sx, sy))]
    return points, labels, box


def prepare(archive, out, edition, limit):
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    digest = models.tuned_decoder_digest()
    if source["base_decoder_sha256"] != digest:
        raise ValueError("既有训练样本清单与当前 SAM 微调权重不匹配。")
    root = out / "cache"
    root.mkdir(parents=True, exist_ok=True)
    record_path = root / "manifest.json"
    if record_path.exists():
        raise FileExistsError(record_path)
    engine = models.predictor("tuned")
    manifest = {
        "archive": str(archive),
        "edition": edition,
        "seed": SEED,
        "base_decoder_sha256": digest,
        "patch_size": PATCH,
    }
    with ZipFile(archive) as data:
        for split in ("train", "dev"):
            rng = np.random.default_rng(SEED if split == "train" else SEED + 1)
            entries, total, skipped = [], 0, 0
            directory = root / split
            directory.mkdir(exist_ok=True)
            records = source[split]["frames"][:limit]
            for i, record in enumerate(records):
                rgb, reduced, low_rgb, target, valid, overlap = frame_from_archive(
                    data, record, edition
                )
                engine.set_image(reduced)
                h, w = target.shape
                patches = []
                for points in record["prompts"].values():
                    initial = models.prompt_prediction(engine, *points)
                    coarse = cv2.resize(initial, (w, h), interpolation=cv2.INTER_LINEAR)
                    bounds = refinement_bounds(
                        coarse, *scaled_prompt(points, reduced.shape[:2], target.shape)
                    )
                    if bounds is None or (
                        split == "train" and scores(coarse > 0, target, valid)[0] < 0.45
                    ):
                        skipped += 1
                        continue
                    for window, weights in training_windows(target, valid, bounds, rng):
                        patches.append(
                            (
                                rgb[window],
                                low_rgb[window],
                                coarse[window],
                                target[window],
                                weights,
                            )
                        )
                if patches:
                    path = directory / f"{i:04d}.npz"
                    full, low, logits, truth, weights = zip(*patches)
                    np.savez_compressed(
                        path,
                        full=np.stack(full),
                        low=np.stack(low),
                        coarse=np.stack(logits).astype(np.float16),
                        truth=np.stack(truth),
                        weights=np.stack(weights).astype(np.uint8),
                    )
                    entries.append(
                        {
                            "cache": path.relative_to(root).as_posix(),
                            "image": record["image"],
                            "instance": record["object"],
                            "full_width": w,
                            "full_height": h,
                            "low_height": reduced.shape[0],
                            "label_pair_iou": overlap,
                            "patches": len(patches),
                        }
                    )
                    total += len(patches)
                if (i + 1) % 50 == 0:
                    print(f"{split}: {i + 1}/{len(records)}", flush=True)
            if not total:
                raise ValueError(f"{split} 未生成任何配对的裁剪。")
            manifest[split] = {
                "frames": entries,
                "patches": total,
                "skipped_prompts": skipped,
            }
            print(
                f"{split}: {len(entries)} frames, {total} paired patches, {skipped} skipped prompts",
                flush=True,
            )
    if models.tuned_decoder_digest() != digest:
        raise RuntimeError("生成缓存期间基础权重发生变化。")
    record_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def patch_array(out, split, manifest):
    group = manifest[split]
    n, size = group["patches"], manifest["patch_size"]
    color_shape, label_shape = (n, size, size, 3), (n, size, size)
    arrays = {
        "full": np.empty(color_shape, np.uint8),
        "low": np.empty(color_shape, np.uint8),
        "coarse": np.empty(label_shape, np.float16),
        "truth": np.empty(label_shape, bool),
        "weights": np.empty(label_shape, np.uint8),
    }
    offset = 0
    for row in group["frames"]:
        count = row["patches"]
        with np.load(out / "cache" / row["cache"]) as data:
            for key in arrays:
                arrays[key][offset : offset + count] = data[key]
        offset += count
    if offset != n:
        raise ValueError("裁剪缓存条数不一致。")
    return arrays


def tensors(array, indices, device, flip):
    result = {}
    for name in ("full", "low"):
        result[name] = (
            torch.from_numpy(array[name][indices]).permute(0, 3, 1, 2).to(device)
        )
    for name in ("coarse", "truth", "weights"):
        result[name] = torch.from_numpy(array[name][indices])[:, None].to(
            device, torch.float32
        )
    return {k: x.flip(-1) for k, x in result.items()} if flip else result


@torch.inference_mode()
def validation(networks, patches, device):
    totals = {name: [] for name in networks}
    for net in networks.values():
        net.eval()
    for start in range(0, len(patches["truth"]), BATCH):
        batch = tensors(patches, slice(start, start + BATCH), device, False)
        for name, key in zip(CONDITIONS, ("full", "low")):
            mask = (networks[name](batch[key], batch["coarse"])[:, 0] > 0).cpu().numpy()
            for gt, pred, weights in zip(
                patches["truth"][start : start + BATCH],
                mask,
                patches["weights"][start : start + BATCH],
            ):
                totals[name].append(
                    db_eval_boundary(gt, pred, weights == 0, bound_th=2)
                )
    return {name: mean(values) for name, values in totals.items()}


def train(out):
    config = json.loads((out / "cache/manifest.json").read_text(encoding="utf-8"))
    digest = models.tuned_decoder_digest()
    if config["base_decoder_sha256"] != digest:
        raise ValueError("缓存与当前 SAM 微调权重不匹配。")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = models.device()
    full = PixelRefiner().to(device)
    networks = {"full": full, "480p": copy.deepcopy(full)}
    optimizers = {
        name: torch.optim.AdamW(net.parameters(), lr=3e-4)
        for name, net in networks.items()
    }
    with (out / "history.csv").open("x", newline="", encoding="utf-8") as file:
        (out / "training.json").write_text(
            json.dumps(
                {
                    "conditions": list(CONDITIONS),
                    "epochs_per_condition": EPOCHS,
                    "initialization": "shared U-Net state dict",
                    "seed": SEED,
                    "patch_size": config["patch_size"],
                    "batch": BATCH,
                    "optimizer": "AdamW",
                    "learning_rate": 3e-4,
                    "horizontal_flip_probability": 0.5,
                    "loss": "same boundary/thin-weighted BCE plus Dice",
                    "train_patches": config["train"]["patches"],
                    "dev_patches": config["dev"]["patches"],
                    "base_decoder_sha256": digest,
                    "train_targets": "full-resolution DAVIS annotation in both conditions",
                    "change": "downsample RGB to matching 480p then bilinear upsample in 480p condition",
                    "device": device,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        writer = csv.writer(file)
        writer.writerow(("epoch", "input", "loss", "dev_patch_f2", "seconds"))
        patches = patch_array(out, "train", config)
        dev = patch_array(out, "dev", config)
        n = len(patches["truth"])
        for epoch in range(1, EPOCHS + 1):
            start = perf_counter()
            for net in networks.values():
                net.train()
            order = rng.permutation(n)
            totals = {name: torch.zeros((), device=device) for name in networks}
            for i in range(0, n, BATCH):
                keys = order[i : i + BATCH]
                batch = tensors(patches, keys, device, rng.random() < 0.5)
                for name, key in zip(CONDITIONS, ("full", "low")):
                    optimizer = optimizers[name]
                    optimizer.zero_grad(set_to_none=True)
                    loss = patch_loss(
                        networks[name](batch[key], batch["coarse"]),
                        batch["truth"],
                        batch["weights"],
                    )
                    loss.backward()
                    optimizer.step()
                    totals[name] += loss.detach() * len(keys)
            dev_f2 = validation(networks, dev, device)
            for name in CONDITIONS:
                row = [
                    epoch,
                    name,
                    float(totals[name]) / n,
                    dev_f2[name],
                    perf_counter() - start,
                ]
                writer.writerow(row)
                print(row, flush=True)
            file.flush()
        for name in CONDITIONS:
            path = out / f"epoch-{EPOCHS}-{name}.pt"
            temp = path.with_suffix(".pt.part")
            torch.save(
                {
                    "refiner": networks[name].state_dict(),
                    "use_rgb": True,
                    "epoch": EPOCHS,
                    "base_decoder_sha256": digest,
                    "input": name,
                },
                temp,
            )
            temp.replace(path)


def metrics(target, mask, valid, thin):
    # Boundary tolerances below are absolute pixels, so omit empty background.
    x, y, w, h = cv2.boundingRect(((target | mask) & valid).astype(np.uint8))
    if w and h:
        image_h, image_w = target.shape
        region = np.s_[
            max(0, y - 3) : min(image_h, y + h + 3),
            max(0, x - 3) : min(image_w, x + w + 3),
        ]
        target, mask, valid, thin = [arr[region] for arr in (target, mask, valid, thin)]
    iou, _ = scores(mask, target, valid)
    thin_count = int(thin.sum())
    void = ~valid
    return {
        "iou": iou,
        "f_at_1px": db_eval_boundary(target, mask, void, bound_th=1),
        "f_at_2px": db_eval_boundary(target, mask, void, bound_th=2),
        "thin_recall": float((mask & thin).sum() / thin_count) if thin_count else None,
        "thin_pixels": thin_count,
    }


def wheel_metrics(target, valid, thin, mask, window):
    # Match full-frame contours first, then count within the wheel crop.
    from davis2017.metrics import _seg2bmap

    gt = _seg2bmap(target & valid)
    pred = _seg2bmap(mask & valid)
    disk = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    counts = (gt[window].sum(), pred[window].sum())
    if all(counts):
        recall = (gt & (cv2.dilate(pred.astype(np.uint8), disk) > 0))[
            window
        ].sum() / counts[0]
        precision = (pred & (cv2.dilate(gt.astype(np.uint8), disk) > 0))[
            window
        ].sum() / counts[1]
        f2 = 2 * recall * precision / (recall + precision) if recall + precision else 0
    else:
        f2 = float(counts[0] == counts[1])
    thin = thin[window]
    return {
        "local_f2": float(f2),
        "thin_pixels": int(thin.sum()),
        "thin_preserved": int((thin & mask[window]).sum()),
    }


def evaluate(archive, out, edition, limit):
    rows, wheel, metadata = [], {}, None
    digest = models.tuned_decoder_digest()
    nets = {
        name: load_refiner(out / f"epoch-{EPOCHS}-{name}.pt", digest)[0]
        for name in CONDITIONS
    }
    engine = models.predictor("tuned")
    records = json.loads(PROMPTS.read_text(encoding="utf-8"))[:limit]
    with ZipFile(archive) as data:
        for i, record in enumerate(records):
            path = Path(record["image"])
            rgb, low, low_rgb, target, valid, _ = frame_from_archive(
                data, record, edition
            )
            thin = thin_pixels(target, valid)
            h, w = target.shape
            lh, lw = low.shape[:2]
            engine.set_image(low)
            is_wheel = path.parent.name == "bike-packing" and path.stem == "00000"
            if is_wheel:
                rect = [
                    round(v * k)
                    for v, k in zip(WHEEL_ROI, (w / lw, h / lh, w / lw, h / lh))
                ]
                a, b, c, d = rect
                window = np.s_[b:d, a:c]
                wheel.update(
                    rgb=rgb[window],
                    low=low_rgb[window],
                    truth=target[window],
                    valid=valid[window],
                    thin=thin[window],
                )
                metadata = {
                    "image": record["image"],
                    "object": record["object"],
                    "full_size": [w, h],
                    "low_size": [lw, lh],
                    "roi": rect,
                    "prompts": {},
                    "metrics": {},
                }
            for mode, prompt in record["prompts"].items():
                initial = models.prompt_prediction(engine, *prompt)
                coarse = cv2.resize(initial, (w, h), interpolation=cv2.INTER_LINEAR)
                bounds = refinement_bounds(
                    coarse, *scaled_prompt(prompt, low.shape[:2], target.shape)
                )
                masks = {"coarse": coarse > 0}
                for name, pixels in zip(CONDITIONS, (rgb, low_rgb)):
                    masks[name] = refine_pixels(nets[name], pixels, coarse, bounds) > 0
                for name, mask in masks.items():
                    rows.append(
                        {
                            "video": path.parent.name,
                            "frame": path.stem,
                            "object": record["object"],
                            "prompt": mode,
                            "input": name,
                            **metrics(target, mask, valid, thin),
                        }
                    )
                if is_wheel and mode in ("point", "box"):
                    metadata["prompts"][mode] = scaled_prompt(
                        prompt, low.shape[:2], target.shape
                    )
                    metadata["metrics"][mode] = {}
                    for name, mask in masks.items():
                        wheel[f"{mode}_{name}"] = mask[window]
                        metadata["metrics"][mode][name] = wheel_metrics(
                            target, valid, thin, mask, window
                        )
            if (i + 1) % 15 == 0:
                print(f"eval {i + 1}/{len(records)} full frames", flush=True)
    output = out / "evaluation"
    output.mkdir(exist_ok=True)
    with (output / "per-prompt.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    group = defaultdict(lambda: defaultdict(list))
    for row in rows:
        group[row["input"], row["prompt"]][row["video"]].append(row)
    summary = defaultdict(dict)
    for (name, mode), videos in group.items():
        result = {}
        for key in ("iou", "f_at_1px", "f_at_2px", "thin_recall"):
            values = [
                [r[key] for r in rs if r[key] is not None] for rs in videos.values()
            ]
            result[key] = mean(mean(v) for v in values if v)
        summary[name][mode] = result
    (output / "summary.json").write_text(
        json.dumps(
            {
                "frames": len(records),
                "videos": len({r["video"] for r in rows}),
                "train_epochs_each": EPOCHS,
                "reference": "official full-resolution mask",
                "scores": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if metadata:
        np.savez_compressed(output / "wheel.npz", **wheel)
        (output / "wheel.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    print(f"{len(rows)} measurements saved to {output}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "train", "evaluate"))
    p.add_argument("--archive", type=Path, default=ARCHIVE)
    p.add_argument("--out", type=Path, default=OUTPUT)
    p.add_argument(
        "--edition", default="Full-Resolution", help="DAVIS 子目录，480p 仅用于冒烟测试"
    )
    p.add_argument("--limit", type=int, help="冒烟测试的帧数上限")
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error("帧数上限必须为正数。")
    if args.stage == "prepare":
        prepare(args.archive, args.out, args.edition, args.limit)
    elif args.stage == "train":
        train(args.out)
    else:
        evaluate(args.archive, args.out, args.edition, args.limit)


if __name__ == "__main__":
    main()
