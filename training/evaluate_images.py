"""Paired, video-balanced evaluation with identical stored prompts."""

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import median

import numpy as np
import torch
from PIL import Image

from sam_app import models
from training.data import DATA, SPLIT, prompt, read_target, samples, scores
from training.metrics import db_eval_boundary


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(engine, records, version, output, measure_boundary=False):
    values = {}
    decode_times = []
    with (output / (version + ".csv")).open("w", newline="") as f:
        writer = csv.writer(f)
        columns = [
            "video",
            "frame",
            "object",
            "prompt",
            "iou",
            "dice",
            "encode_ms",
            "decode_ms",
        ]
        writer.writerow(columns + (["boundary_f"] if measure_boundary else []))
        for index, record in enumerate(records):
            with Image.open(DATA / record["image"]) as img:
                rgb = np.array(img.convert("RGB"))
            with Image.open(DATA / record["mask"]) as img:
                ids = np.array(img)
            target, valid = ids == record["object"], ids != 255
            if measure_boundary and index == 0:
                engine.set_image(rgb)
                for p in record["prompts"].values():
                    models.mask_from_prompt(engine, *p)
            sync()
            begin = time.perf_counter()
            engine.set_image(rgb)
            sync()
            enc = 1000 * (time.perf_counter() - begin)
            for mode, p in record["prompts"].items():
                begin = time.perf_counter()
                mask = models.mask_from_prompt(engine, *p)
                sync()
                dec = 1000 * (time.perf_counter() - begin)
                quality = list(scores(mask, target, valid))
                boundary = (
                    [db_eval_boundary(target, mask, ~valid)] if measure_boundary else []
                )
                video = Path(record["image"]).parent.name
                values.setdefault(mode, {}).setdefault(video, []).append(
                    quality + boundary
                )
                decode_times.append(dec)
                writer.writerow(
                    [
                        video,
                        Path(record["image"]).stem,
                        record["object"],
                        mode,
                        *quality,
                        enc,
                        dec,
                        *boundary,
                    ]
                )
    metrics = {
        mode: np.mean([np.mean(a, axis=0) for a in groups.values()], axis=0).tolist()
        for mode, groups in values.items()
    }
    return metrics, median(decode_times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "eval"], default="dev")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--include-refiner",
        action="store_true",
        help="对比原版、微调版和边界细化版，并记录边界 F 和解码耗时",
    )
    parser.add_argument("--refiner-checkpoint", type=Path, default=models.REFINED)
    args = parser.parse_args()
    if args.out is None:
        name = "refinement-evaluation" if args.include_refiner else "paired-evaluation"
        args.out = models.ROOT / "outputs" / name
    if args.include_refiner and not args.refiner_checkpoint.is_file():
        parser.error("请先训练边界细化分支。")
    if args.include_refiner and (args.out / "summary.json").exists():
        parser.error("该输出目录已有评测结果，请选择新的 --out。")
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(77)
    rows = []
    splits = json.loads(SPLIT.read_text())
    for pair in samples(DATA, splits[args.split], args.frames):
        item = read_target(pair, rng)
        if item is None:
            continue
        _, mask, _, obj = item
        prompts = {
            label: prompt(mask, rng, mode, jitter)
            for label, mode, jitter in (
                ("point", "point", 0),
                ("box", "box", 0),
                ("box10", "box", 0.1),
                ("box20", "box", 0.2),
            )
        }
        rows.append(
            {
                "image": pair[0].relative_to(DATA).as_posix(),
                "mask": pair[1].relative_to(DATA).as_posix(),
                "object": obj,
                "prompts": prompts,
            }
        )
    if not rows:
        parser.error("评测集没有可见实例。")
    (args.out / "prompts.json").write_text(json.dumps(rows, indent=2))
    report = {"split": args.split, "frames": len(rows), "metrics": ["IoU", "Dice"]}
    versions = ["original", "tuned"]
    if args.include_refiner:
        versions.append("refined")
        report["metrics"].append("boundary_f")
        report["decode_ms_median"] = {}
    for version in versions:
        engine = models.predictor(version, refiner_checkpoint=args.refiner_checkpoint)
        report[version], decode_ms = benchmark(
            engine, rows, version, args.out, measure_boundary=args.include_refiner
        )
        if args.include_refiner:
            report["decode_ms_median"][version] = decode_ms
        del engine
    (args.out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
