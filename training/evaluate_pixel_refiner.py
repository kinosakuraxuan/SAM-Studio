"""Compare both native-pixel heads to the two existing SAM variants."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import torch

from sam_app import models
from sam_app.pixel_refiner import PixelRefiner, refine_pixels, refinement_bounds
from training.data import scores
from training.metrics import db_eval_boundary
from training.prepare_pixel_refiner import DEFAULT_OUT
from training.train_pixel_refiner import read_eval_frame

METRICS = ("iou", "dice", "boundary_f", "f_at_1px", "f_at_2px", "thin_recall")


def load_refiner(path, digest):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["base_decoder_sha256"] != digest:
        raise ValueError(f"细化权重与当前微调解码器不匹配：{path}")
    model = PixelRefiner(use_rgb=checkpoint["use_rgb"])
    model.load_state_dict(checkpoint["refiner"])
    return model.eval().to(models.device()), checkpoint["epoch"]


def record_metrics(
    version, sequence, frame, instance, mode, mask, truth, extra_ms=None
):
    target, valid, thin = truth
    iou, dice = scores(mask, target, valid)
    void = ~valid
    thin_count = int(thin.sum())
    return {
        "version": version,
        "video": sequence,
        "frame": frame,
        "object": instance,
        "prompt": mode,
        "iou": iou,
        "dice": dice,
        "boundary_f": db_eval_boundary(target, mask, void),
        "f_at_1px": db_eval_boundary(target, mask, void, bound_th=1),
        "f_at_2px": db_eval_boundary(target, mask, void, bound_th=2),
        "thin_recall": float((mask & thin).sum() / thin_count) if thin_count else None,
        "thin_pixels": thin_count,
        "refinement_ms": extra_ms,
    }


def evaluate(records, left, refiners):
    rows = []
    engine = models.predictor(left)
    for index, record in enumerate(records):
        path = Path(record["image"])
        rgb, *truth = read_eval_frame(record)
        engine.set_image(rgb)
        for mode, p in record["prompts"].items():
            coarse = models.prompt_prediction(engine, *p)
            key = path.parent.name, path.stem, record["object"], mode
            rows.append(record_metrics(left, *key, coarse > 0, truth))
            if refiners:
                bounds = refinement_bounds(coarse, *p)
                for name, network in refiners.items():
                    start = perf_counter()
                    mask = refine_pixels(network, rgb, coarse, bounds) > 0
                    extra_ms = (perf_counter() - start) * 1000
                    rows.append(record_metrics(name, *key, mask, truth, extra_ms))
        if (index + 1) % 30 == 0:
            print(f"{left}: {index + 1}/{len(records)} frames evaluated", flush=True)
    return rows


def summarize(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[r["version"], r["prompt"]][r["video"]].append(r)
    output = defaultdict(dict)
    for (version, prompt), videos in groups.items():
        values = {}
        for metric in METRICS:
            means = []
            for sequence in videos.values():
                measured = [v[metric] for v in sequence if v[metric] is not None]
                if measured:
                    means.append(mean(measured))
            values[metric] = mean(means) if means else None
        durations = [
            r["refinement_ms"]
            for seq in videos.values()
            for r in seq
            if r["refinement_ms"] is not None
        ]
        if durations:
            values["refinement_ms_median"] = median(durations)
        output[version][prompt] = values
    return dict(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=models.ROOT / "outputs/refinement-evaluation/prompts.json",
    )
    args = parser.parse_args()
    report = args.report or args.out / "evaluation"
    summary_path = report / "summary.json"
    if summary_path.exists():
        parser.error("该目录已有评测结果。")
    records = json.loads(args.prompts.read_text(encoding="utf-8"))
    if not records:
        parser.error("没有可评测的提示。")
    report.mkdir(parents=True, exist_ok=True)
    digest = models.tuned_decoder_digest()
    refiners, epochs = {}, {}
    for name in ("no-rgb", "rgb"):
        refiners[name], epochs[name] = load_refiner(
            args.out / f"best-{name}.pt", digest
        )
    rows = evaluate(records, "tuned", refiners)
    del refiners
    rows.extend(evaluate(records, "refined", {}))
    with (report / "per-prompt.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "frames": len(records),
        "videos": len({r["video"] for r in rows}),
        "prompts_source": str(args.prompts),
        "selected_epochs": epochs,
        "same_fixed_prompts": True,
        "thin_metric": "foreground removed by 5x5 ellipse opening",
        "scores": summarize(rows),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} paired measurements to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
