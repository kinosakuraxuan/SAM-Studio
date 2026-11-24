"""Train a native-resolution U-Net and its matched no-RGB ablation."""

import argparse
import copy
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from sam_app import models
from sam_app.pixel_refiner import PixelRefiner, refine_pixels, refinement_bounds
from training.data import DATA, scores
from training.metrics import db_eval_boundary
from training.prepare_pixel_refiner import DEFAULT_OUT, SEED, thin_pixels


def read_training_patches(cache, info, size):
    n = info["patches"]
    patches = {
        "rgb": np.empty((n, size, size, 3), dtype=np.uint8),
        "coarse": np.empty((n, size, size), dtype=np.float16),
        "truth": np.empty((n, size, size), dtype=bool),
        "weights": np.empty((n, size, size), dtype=np.uint8),
    }
    start = 0
    for frame in info["frames"]:
        with np.load(cache / frame["cache"]) as entry:
            count = len(entry["truth"])
            for name, dst in patches.items():
                dst[start : start + count] = entry[name]
            start += count
    if start != n:
        raise ValueError("缓存裁剪数与清单不一致。")
    return patches


def read_eval_frame(record):
    with Image.open(DATA / record["image"]) as image:
        rgb = np.array(image.convert("RGB"))
    with Image.open(DATA / record["mask"]) as image:
        ids = np.array(image)
    target, valid = ids == record["object"], ids != 255
    return rgb, target, valid, thin_pixels(target, valid)


def patch_loss(logits, truth, weights):
    bce = F.binary_cross_entropy_with_logits(logits, truth, reduction="none")
    valid = weights > 0
    probability = logits.sigmoid() * valid
    truth = truth * valid
    dice = (2 * (probability * truth).sum() + 1) / (probability.sum() + truth.sum() + 1)
    return (bce * weights).sum() / weights.sum().clamp_min(1) + 1 - dice


def dev_scores(networks, cache, dev):
    summaries = {name: [] for name in networks}
    for network in networks.values():
        network.eval()
    for record in dev:
        rgb, target, valid, _ = read_eval_frame(record)
        with np.load(cache / record["cache"]) as logits:
            for mode, points in record["prompts"].items():
                coarse = logits[mode]
                bounds = refinement_bounds(coarse, *points)
                for name, network in networks.items():
                    mask = refine_pixels(network, rgb, coarse, bounds) > 0
                    iou, _ = scores(mask, target, valid)
                    f2 = db_eval_boundary(target, mask, ~valid, bound_th=2)
                    summaries[name].append((iou, f2))
    return {name: np.mean(rows, axis=0).tolist() for name, rows in summaries.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    if min(args.epochs, args.batch) < 1:
        parser.error("轮次和批大小必须为正数。")
    if (args.out / "history.csv").exists():
        parser.error("该目录已有训练日志。")
    cache = args.out / "cache"
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    digest = models.tuned_decoder_digest()
    if manifest["base_decoder_sha256"] != digest:
        raise ValueError("粗掩膜缓存与当前微调解码器不匹配。")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = models.device()
    patches = read_training_patches(cache, manifest["train"], manifest["patch_size"])
    refiner = PixelRefiner().to(device)
    ablation = copy.deepcopy(refiner)
    ablation.use_rgb = False
    networks = {"rgb": refiner, "no-rgb": ablation}
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=3e-4)
        for name, model in networks.items()
    }
    best = {name: -1.0 for name in networks}
    info = {
        "epochs": args.epochs,
        "batch": args.batch,
        "seed": SEED,
        "device": device,
        "parameters_per_model": sum(p.numel() for p in refiner.parameters()),
        "base_decoder_sha256": digest,
        "patches": len(patches["truth"]),
        "dev_frames": len(manifest["dev"]["frames"]),
        "selection_metric": "dev_boundary_f2",
    }
    (args.out / "training.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    print(info, flush=True)
    with (args.out / "history.csv").open("x", encoding="utf-8", newline="") as file:
        log = csv.writer(file)
        log.writerow(["epoch", "version", "loss", "dev_iou", "dev_f2", "seconds"])
        for epoch in range(1, args.epochs + 1):
            start = perf_counter()
            for network in networks.values():
                network.train()
            totals = {name: torch.zeros((), device=device) for name in networks}
            order = rng.permutation(len(patches["truth"]))
            for offset in range(0, len(order), args.batch):
                indices = order[offset : offset + args.batch]
                rgb = (
                    torch.from_numpy(patches["rgb"][indices])
                    .permute(0, 3, 1, 2)
                    .to(device)
                )
                coarse, truth, weights = [
                    torch.from_numpy(patches[name][indices])[:, None].to(
                        device, torch.float32
                    )
                    for name in ("coarse", "truth", "weights")
                ]
                if rng.random() < 0.5:
                    rgb, coarse, truth, weights = [
                        tensor.flip(-1) for tensor in (rgb, coarse, truth, weights)
                    ]
                for name, network in networks.items():
                    optimizer = optimizers[name]
                    optimizer.zero_grad(set_to_none=True)
                    loss = patch_loss(network(rgb, coarse), truth, weights)
                    loss.backward()
                    optimizer.step()
                    totals[name] += loss.detach() * len(indices)
            validation = dev_scores(networks, cache, manifest["dev"]["frames"])
            for name, (iou, f2) in validation.items():
                row = [
                    epoch,
                    name,
                    float(totals[name]) / len(order),
                    iou,
                    f2,
                    perf_counter() - start,
                ]
                log.writerow(row)
                print(row, flush=True)
                if f2 > best[name]:
                    best[name] = f2
                    target = args.out / f"best-{name}.pt"
                    temp = target.with_suffix(".pt.part")
                    torch.save(
                        {
                            "refiner": networks[name].state_dict(),
                            "use_rgb": networks[name].use_rgb,
                            "epoch": epoch,
                            "dev_f2": f2,
                            "base_decoder_sha256": digest,
                        },
                        temp,
                    )
                    temp.replace(target)
            file.flush()


if __name__ == "__main__":
    main()
