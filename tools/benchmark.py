"""Time independent CPU/GPU synchronous image interactions on one DAVIS frame."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from PIL import Image

from sam_app import models
from training.data import DATA


def timed(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = perf_counter()
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (perf_counter() - t) * 1000


def benchmark_cache(rgb, box, repeat):
    t = perf_counter()
    engine = models.predictor()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_ms = (perf_counter() - t) * 1000

    def full():
        engine.set_image(rgb)
        models.mask_from_prompt(engine, [], [], box)

    def cached():
        models.mask_from_prompt(engine, [], [], box)

    full()
    cached()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cold = [timed(full) for _ in range(repeat)]
    warm = [timed(cached) for _ in range(repeat)]
    return {
        "load_ms": load_ms,
        "reencode_and_decode_ms_median": median(cold),
        "cached_decode_ms_median": median(warm),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20
        if torch.cuda.is_available()
        else None,
    }


def compare_refiner(rgb, box, repeat):
    engines = {version: models.predictor(version) for version in ("tuned", "refined")}
    for engine in engines.values():
        engine.set_image(rgb)
        for _ in range(8):
            models.mask_from_prompt(engine, [], [], box)
    times = {version: [] for version in engines}
    for i in range(repeat):
        order = tuple(engines) if i % 2 == 0 else tuple(reversed(engines))
        for version in order:
            engine = engines[version]
            times[version].append(
                timed(lambda: models.mask_from_prompt(engine, [], [], box))
            )
    return {
        "decode_ms_median": {
            version: median(values) for version, values in times.items()
        },
        "added_ms_paired_median": median(
            refined - tuned for tuned, refined in zip(times["tuned"], times["refined"])
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=12)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compare-refiner", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("重复次数必须为正数")
    if args.out is None:
        name = (
            "refinement-evaluation/paired_latency.json"
            if args.compare_refiner
            else "performance.json"
        )
        args.out = models.ROOT / "outputs" / name
    with Image.open(DATA / "JPEGImages/480p/car-turn/00000.jpg") as img:
        rgb = np.array(img.convert("RGB"))
    box = [700, 212, 800, 304]
    benchmark = compare_refiner if args.compare_refiner else benchmark_cache
    result = {
        "device": models.device(),
        "repeat": args.repeat,
        "source": "car-turn/00000",
        "torch": str(torch.__version__),
        **benchmark(rgb, box, args.repeat),
    }
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
