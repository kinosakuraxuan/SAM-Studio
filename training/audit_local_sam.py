"""Fixed-field DAVIS label audit and paired local SAM re-encoding, without training."""

import csv
import json
from itertools import pairwise
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import torch
from PIL import Image

from sam_app import models
from training.compare_davis_resolution import (
    ARCHIVE,
    OUTPUT,
    PROMPTS,
    SOURCE,
    frame_from_archive,
    scaled_prompt,
    wheel_metrics,
)

OUT = models.ROOT / "outputs/wheel-crop-audit"
BRANCHES = ("scene", "crop480", "cropfull")
# Centres selected on RGB overview frames mapped to a 320 x 180 coordinate grid.
# Three specific frames per sequence; the selected instance is inherited from SOURCE.
FIELDS = {
    "stroller": [("00000", 112, 56), ("00030", 120, 41), ("00070", 116, 46)],
    "motorbike": [("00000", 104, 96), ("00023", 176, 100), ("00037", 277, 96)],
    "schoolgirls": [("00008", 215, 94), ("00043", 237, 112), ("00079", 96, 84)],
    "hockey": [("00008", 140, 77), ("00016", 141, 76), ("00032", 149, 98)],
    "swing": [("00013", 196, 50), ("00026", 206, 27), ("00045", 174, 40)],
    "tennis": [("00000", 146, 88), ("00030", 138, 78), ("00061", 158, 87)],
    "bmx-bumps": [("00000", 216, 128), ("00044", 9, 112), ("00089", 219, 113)],
    "cat-girl": [("00000", 169, 111), ("00044", 133, 39), ("00088", 174, 78)],
    "tuk-tuk": [("00000", 201, 30), ("00029", 306, 60), ("00058", 205, 22)],
    "dog-agility": [("00000", 269, 111), ("00012", 231, 91), ("00024", 125, 81)],
}
# Coordinates within the previously published 536 x 759 wheel field, right/bottom exclusive.
DETAILS = {"A": (138, 225, 253, 261, 240), "B": (160, 442, 283, 499, 465)}


def save_json(file, data):
    file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def region(rect):
    x, y, r, b = rect
    return np.s_[y:b, x:r]


def clamp_box(rect, w, h):
    x, y, r, b = rect
    return max(0, x), max(0, y), min(w, r), min(h, b)


def sample_set():
    original = json.loads(PROMPTS.read_text(encoding="utf-8"))
    wheel = next(r for r in original if r["image"].endswith("/bike-packing/00000.jpg"))
    previous = json.loads(
        (OUTPUT / "evaluation/wheel.json").read_text(encoding="utf-8")
    )
    assert wheel["object"] == previous["object"]
    yield "heldout", wheel, previous["roi"], None
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    records = {
        r["image"]: (split, r)
        for split in ("train", "dev")
        for r in source[split]["frames"]
    }
    for video, entries in FIELDS.items():
        for frame, x, y in entries:
            split, record = records[f"JPEGImages/480p/{video}/{frame}.jpg"]
            yield split, record, None, (x, y)


def runs(line):
    return np.flatnonzero(np.diff(np.r_[False, line, False])).reshape(-1, 2)


def spoke_audit(target, valid, masks):
    detail = {}
    for code, (x, y, r, b, cut) in DETAILS.items():
        patch = np.s_[y:b, x:r]
        assert valid[patch].all()
        gt = target[patch]
        intervals = runs(target[cut, x:r]) + x
        assert len(intervals) >= 2 and intervals[0, 0] > x and intervals[-1, 1] < r
        gaps = [(int(i[1]), int(j[0])) for i, j in pairwise(intervals)]
        record = {
            "rect": [x, y, r, b],
            "cut_y": cut,
            "foreground_runs": intervals.tolist(),
            "interior_gap_runs": gaps,
            "fg_pixels": int(gt.sum()),
            "bg_pixels": int((~gt).sum()),
            "scores": {},
        }
        for name, mask in masks.items():
            pred = mask[patch]
            record["scores"][name] = {
                "spoke_tp": int((pred & gt).sum()),
                "gap_fp": int((pred & ~gt).sum()),
                "fg_run_hit_pixels": [int(mask[cut, i:j].sum()) for i, j in intervals],
                "gap_run_fp_pixels": [int(mask[cut, i:j].sum()) for i, j in gaps],
            }
        detail[code] = record
    return detail


def mask_prompt(logits):
    return cv2.resize(logits, (256, 256), interpolation=cv2.INTER_LINEAR)[None].clip(
        -32, 32
    )


@torch.inference_mode()
def main():
    OUT.mkdir(exist_ok=True)
    fields = OUT / "fields"
    fields.mkdir(exist_ok=True)
    cv2.setNumThreads(2)
    engine = models.predictor("tuned")
    digest = models.tuned_decoder_digest()
    rows, cases = [], []
    with ZipFile(ARCHIVE) as archive:
        for split, rec, fixed, centre in sample_set():
            path = Path(rec["image"])
            key = f"{path.parent.name}-{path.stem}"
            rgb, small, low, truth, valid, _ = frame_from_archive(
                archive, rec, "Full-Resolution"
            )
            h, w = truth.shape
            if fixed:
                box = tuple(fixed)
            else:
                x, y = (round(centre[0] * w / 320), round(centre[1] * h / 180))
                box = clamp_box((x - 96, y - 96, x + 96, y + 96), w, h)
            x, y, r, b = box
            context = clamp_box((x - 64, y - 64, r + 64, b + 64), w, h)
            a, c, _, _ = context
            crop = region(context)
            inner = region((x - a, y - c, r - a, b - c))
            field = region(box)
            case = {
                "key": key,
                "split": split,
                "object": rec["object"],
                "native_wh": [w, h],
                "low_wh": [small.shape[1], small.shape[0]],
                "field": box,
                "context": context,
                "source_prompts": {},
                "native_prompts": {},
            }
            engine.set_image(rgb)
            initial, priors, masks = {}, {}, {}
            for mode in ("point", "box"):
                cue = rec["prompts"][mode]
                prompt = scaled_prompt(cue, small.shape[:2], (h, w))
                case["source_prompts"][mode] = cue
                case["native_prompts"][mode] = prompt
                logits = models.prompt_prediction(engine, *prompt)
                initial[mode] = logits[crop].copy()
                priors[mode] = mask_prompt(initial[mode])
                masks[f"{mode}_scene"] = initial[mode] > 0
                del logits
            for branch, pixels in (("crop480", low), ("cropfull", rgb)):
                engine.set_image(np.ascontiguousarray(pixels[crop]))
                for mode in ("point", "box"):
                    logits, _, _ = engine.predict(
                        mask_input=priors[mode],
                        multimask_output=False,
                        return_logits=True,
                    )
                    masks[f"{mode}_{branch}"] = logits[0] > 0
            # Full context around the scored field ensures contours at its cut edges are not invented.
            gt, good = truth[crop], valid[crop]
            empty = np.zeros_like(gt)
            for label, pred in masks.items():
                mode, branch = label.split("_")
                f2 = wheel_metrics(gt, good, empty, pred, inner)["local_f2"]
                p, t, v = pred[inner], gt[inner], good[inner]
                rows.append(
                    {
                        "case": key,
                        "split": split,
                        "mode": mode,
                        "branch": branch,
                        "f2": f2,
                        "fg": int((t & v).sum()),
                        "bg": int((~t & v).sum()),
                        "tp": int((p & t & v).sum()),
                        "fp": int((p & ~t & v).sum()),
                    }
                )
            local = {k: v[inner] for k, v in masks.items()}
            if fixed:
                with np.load(OUTPUT / "evaluation/wheel.npz") as prior_data:
                    assert np.array_equal(truth[field], prior_data["truth"])
                    assert np.array_equal(rgb[field], prior_data["rgb"])
                case["transects"] = spoke_audit(truth[field], valid[field], local)
                np.savez_compressed(
                    OUT / "wheel.npz",
                    rgb=rgb[field],
                    low=low[field],
                    truth=truth[field],
                    valid=valid[field],
                    **local,
                )
            else:
                Image.fromarray(rgb[field]).save(fields / f"{key}-rgb.png")
                lab = np.where(valid[field], truth[field].astype(np.uint8) * 255, 127)
                Image.fromarray(lab.astype(np.uint8)).save(fields / f"{key}-label.png")
            cases.append(case)
            print(f"{len(cases):02d}  {key}", flush=True)
    assert digest == models.tuned_decoder_digest()
    save_json(
        OUT / "protocol.json",
        {
            "decoder_sha256": digest,
            "model": "SAM 2.1 Hiera Tiny, tuned decoder",
            "low": "area downsample of same full RGB to official 480p dimensions; bilinear remap before cropping",
            "local_prompt": "same full-frame logits projected to both local mask prompts; no additional clicks or boxes",
            "threshold": 0,
            "context_margin_native_px": 64,
            "cases": cases,
        },
    )
    with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
