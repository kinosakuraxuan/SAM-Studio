"""Evaluate saved local checkpoints on the fixed DAVIS wheel after model selection."""

import json
from zipfile import ZipFile

import cv2
import numpy as np
import torch

from sam_app import models
from training.audit_local_sam import (
    ARCHIVE,
    DETAILS,
    frame_from_archive,
    mask_prompt,
    region,
    sample_set,
    save_json,
    spoke_audit,
    wheel_metrics,
)
from training.audit_local_sam import (
    OUT as PREVIOUS,
)
from training.wheel_samples import OUT

# Native coordinates relative to the established wheel field, marked on RGB before evaluation.
PARTS = {
    "tire": [
        (24, 157),
        (52, 114),
        (76, 90),
        (103, 71),
        (134, 58),
        (157, 53),
        (170, 54),
        (158, 64),
        (137, 68),
        (115, 78),
        (89, 93),
        (67, 113),
        (43, 145),
        (30, 167),
    ],
    "frame": [
        (432, 370),
        (456, 354),
        (467, 359),
        (448, 431),
        (444, 507),
        (447, 611),
        (434, 631),
        (418, 626),
        (413, 576),
        (411, 474),
        (415, 427),
    ],
}


def measurements(gt, valid, prev, pred):
    marked = gt & valid
    gap = ~gt & valid
    tp = int((pred & marked).sum())
    fp = int((pred & gap).sum())
    fg, bg = int(marked.sum()), int(gap.sum())
    return {
        "fg": fg,
        "bg": bg,
        "tp": tp,
        "fp": fp,
        "recovered": int((~prev & pred & marked).sum()),
        "newly_lost": int((prev & ~pred & marked).sum()),
        "iou": tp / (fg + fp) if fg + fp else None,
        "recall": tp / fg if fg else None,
        "gap_fp": fp / bg if bg else None,
    }


@torch.inference_mode()
def main():
    cv2.setNumThreads(2)
    config = json.loads((OUT / "training.json").read_text(encoding="utf-8"))
    protocol = json.loads((PREVIOUS / "protocol.json").read_text(encoding="utf-8"))[
        "cases"
    ][0]
    assert config["base_decoder_sha256"] == models.tuned_decoder_digest()
    assert protocol["key"] == "bike-packing-00000"
    _, record, old_box, _ = next(sample_set())
    with ZipFile(ARCHIVE) as zipfile:
        rgb, _, _, target, valid, _ = frame_from_archive(
            zipfile, record, "Full-Resolution"
        )
    assert tuple(old_box) == tuple(protocol["field"])
    context = region(protocol["context"])
    field = region(old_box)
    x, y, r, b = old_box
    a, c, _, _ = protocol["context"]
    inner = region((x - a, y - c, r - a, b - c))
    engine = models.predictor("tuned")
    engine.set_image(rgb)
    priors = {}
    for mode, cue in protocol["native_prompts"].items():
        logits = models.prompt_prediction(engine, *cue)
        priors[mode] = mask_prompt(logits[context])
    wheel_rgb = rgb[field]
    gt = target[field]
    valid_wheel = valid[field]
    zones = {"wheel": np.ones_like(gt)}
    for key, coords in PARTS.items():
        coverage = np.zeros_like(gt, dtype=np.uint8)
        cv2.fillPoly(coverage, [np.array(coords, np.int32)], 1)
        zones[key] = coverage.astype(bool)
    for name, (x, y, r, b, _) in DETAILS.items():
        zone = np.zeros_like(gt)
        zone[y:b, x:r] = True
        zones[name] = zone
    predictions = {}
    out = {}
    with np.load(PREVIOUS / "wheel.npz") as prior:
        assert np.array_equal(prior["truth"], gt) and np.array_equal(
            prior["rgb"], wheel_rgb
        )
        for arm in ("before", "plain", "weighted"):
            if arm != "before":
                checkpoint = torch.load(
                    OUT / f"{arm}.pt", map_location="cpu", weights_only=True
                )
                assert (
                    checkpoint["base_decoder_sha256"] == config["base_decoder_sha256"]
                )
                engine.model.sam_mask_decoder.load_state_dict(checkpoint["decoder"])
            engine.set_image(np.ascontiguousarray(rgb[context]))
            for mode, guidance in priors.items():
                logits = engine.predict(
                    mask_input=guidance, multimask_output=False, return_logits=True
                )[0][0]
                mask = logits > 0
                key = f"{mode}_{arm}"
                cropped = mask[inner].copy()
                if arm == "before" and not np.array_equal(
                    cropped, prior[f"{mode}_cropfull"]
                ):
                    raise RuntimeError("Pre-training mask differs from prior result")
                predictions[key] = cropped
                out[key] = {
                    zone: measurements(
                        gt, valid_wheel & area, predictions[f"{mode}_before"], cropped
                    )
                    for zone, area in zones.items()
                }
                out[key]["wheel"]["f2"] = wheel_metrics(
                    target[context],
                    valid[context],
                    np.zeros_like(target[context]),
                    mask,
                    inner,
                )["local_f2"]
    np.savez_compressed(
        OUT / "wheel-predictions.npz",
        rgb=wheel_rgb,
        truth=gt,
        valid=valid_wheel,
        tire=zones["tire"],
        frame=zones["frame"],
        **predictions,
    )
    save_json(
        OUT / "wheel-scores.json",
        {
            "chosen": config["chosen"],
            "case": protocol["key"],
            "parts": PARTS,
            "roi": old_box,
            "context": protocol["context"],
            "native_prompts": protocol["native_prompts"],
            "baseline_identical_to_step2": True,
            "scores": out,
            "transects": spoke_audit(gt, valid_wheel, predictions),
        },
    )


if __name__ == "__main__":
    main()
