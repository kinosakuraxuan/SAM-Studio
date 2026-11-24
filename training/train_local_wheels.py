"""Paired local decoder fine-tuning on screened DIS5K wheels with fixed global prompts."""

import copy
import csv
import hashlib
import json
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sam_app import models
from training.audit_local_sam import mask_prompt, region, save_json
from training.data import prompt
from training.metrics import db_eval_boundary
from training.wheel_samples import DATA, FIELDS, OUT, crop_rect, read

REJECT = {"tr13", "tr33", "tr37", "tr40", "tr42", "vd01", "vd10", "vd11"}
ARMS = ("plain", "weighted")
EPOCHS, SEED = 5, 47


def digest(module):
    h = hashlib.sha256()
    for name, t in module.state_dict().items():
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def prepare(engine, manifest_path=OUT / "selected.json"):
    cases = []
    rng = np.random.default_rng(SEED)
    source = {
        r["id"]: r for r in json.loads((DATA / "candidate-files.json").read_text())
    }
    entries = []
    fingerprints = set()
    for key in FIELDS:
        if key in REJECT:
            continue
        rgb, gt = read(key)
        pixel_hash = hashlib.sha256(rgb).hexdigest()
        if pixel_hash in fingerprints:
            raise ValueError("Duplicate photograph")
        fingerprints.add(pixel_hash)
        rect = crop_rect(key, gt.shape)
        window = region(rect)
        engine.set_image(rgb)
        priors, cues = [], []
        for mode in ("point", "box"):
            cue = prompt(gt, rng, mode, jitter=0)
            cues.append(cue)
            priors.append(mask_prompt(models.prompt_prediction(engine, *cue)[window]))
        target = gt[window].copy()
        fg = target.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        thin = fg - cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        gap = (cv2.dilate(thin, kernel) > 0) & ~target
        weights = 1.0 + 3.0 * thin + 2.0 * gap
        cases.append(
            (
                key,
                rgb[window].copy(),
                target,
                np.array(priors),
                weights.astype(np.float32),
            )
        )
        entries.append(
            {
                **source[key],
                "native_size": [rgb.shape[1], rgb.shape[0]],
                "rgb_sha256": pixel_hash,
                "field": rect,
                "point": cues[0],
                "box": cues[1],
            }
        )
    save_json(manifest_path, entries)
    return cases


def backbone(engine, rgb, *, train_encoder=False):
    t = engine._transforms(rgb).unsqueeze(0).to(engine.device)
    with torch.set_grad_enabled(train_encoder and torch.is_grad_enabled()):
        output = engine.model.image_encoder(t)
    # conv_s0/conv_s1 belong to the decoder. Apply each arm's own projections later.
    feats = output["backbone_fpn"][-3:]
    low = feats[-1]
    if engine.model.directly_add_no_mem_embed:
        low = low + engine.model.no_mem_embed.permute(0, 2, 1).unsqueeze(-1)
    return feats[0], feats[1], low


def decode(engine, decoder, features, prior, shape):
    dev = engine.device
    with torch.no_grad():
        sparse, dense = engine.model.sam_prompt_encoder(
            points=None, boxes=None, masks=torch.from_numpy(prior[None]).to(dev)
        )
        pe = engine.model.sam_prompt_encoder.get_dense_pe()
    f0, f1, low = features
    pred, *_ = decoder(
        image_embeddings=low,
        image_pe=pe,
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        repeat_image=False,
        high_res_features=[decoder.conv_s0(f0), decoder.conv_s1(f1)],
    )
    return F.interpolate(pred.float(), shape, mode="bilinear", align_corners=False)[
        0, 0
    ]


def objective(pred, target, weight):
    prob = pred.sigmoid()
    dice = (2 * (prob * target).sum() + 1) / (prob.sum() + target.sum() + 1)
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    return (bce * weight).sum() / weight.sum() + 1 - dice


@torch.no_grad()
def validation(engine, decoders, data):
    rows = {name: [] for name in decoders}
    for _, rgb, gt, priors, _ in data:
        feats = backbone(engine, rgb)
        for name, dec in decoders.items():
            dec.eval()
            for prior in priors:
                mask = (decode(engine, dec, feats, prior, gt.shape) > 0).cpu().numpy()
                iou = float((mask & gt).sum() / (mask | gt).sum())
                f2 = float(db_eval_boundary(gt, mask, bound_th=2))
                rows[name].append((iou, f2))
    return {name: np.mean(values, axis=0).tolist() for name, values in rows.items()}


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    cv2.setNumThreads(2)
    engine = models.predictor("tuned")
    engine.model.eval().requires_grad_(False)
    initial = digest(engine.model.sam_mask_decoder)
    frozen = {
        name: digest(getattr(engine.model, name))
        for name in ("image_encoder", "sam_prompt_encoder")
    }
    samples = prepare(engine)
    train = [s for s in samples if s[0].startswith("tr")]
    dev = [s for s in samples if s[0].startswith("vd")]
    decoders = {
        arm: copy.deepcopy(engine.model.sam_mask_decoder).requires_grad_(True)
        for arm in ARMS
    }
    optim = {
        a: torch.optim.AdamW(d.parameters(), lr=2e-5, weight_decay=0.01)
        for a, d in decoders.items()
    }
    # Compare the differentiable path against the actual predictor before updating.
    _, rgb, gt, priors, _ = dev[0]
    with torch.no_grad():
        engine.set_image(rgb)
        official = engine.predict(
            mask_input=priors[0], multimask_output=False, return_logits=True
        )[0][0]
        check = (
            decode(
                engine, decoders["plain"], backbone(engine, rgb), priors[0], gt.shape
            )
            .cpu()
            .numpy()
        )
        if not np.allclose(official, check, atol=0.001, rtol=0.001):
            raise ValueError(
                f"Decoder replay mismatch, max {np.abs(official - check).max()}"
            )
    baseline = validation(engine, {"before": engine.model.sam_mask_decoder}, dev)[
        "before"
    ]
    best = {a: (-1, 0) for a in ARMS}
    history = []
    for epoch in range(1, EPOCHS + 1):
        start = time.perf_counter()
        for dec in decoders.values():
            dec.train()
        losses = {a: [] for a in ARMS}
        for idx in rng.permutation(len(train)):
            _, rgb, gt, priors, weight = train[idx]
            feats = backbone(engine, rgb)
            target = torch.from_numpy(gt).to(engine.device, torch.float32)
            weighted = torch.from_numpy(weight).to(engine.device)
            for a in ARMS:
                optim[a].zero_grad(set_to_none=True)
                for prior in priors:
                    pred = decode(engine, decoders[a], feats, prior, gt.shape)
                    loss = objective(
                        pred,
                        target,
                        weighted if a == "weighted" else torch.ones_like(target),
                    )
                    (loss / len(priors)).backward()
                    losses[a].append(float(loss.detach()))
                torch.nn.utils.clip_grad_norm_(decoders[a].parameters(), 1.0)
                optim[a].step()
        val = validation(engine, decoders, dev)
        for a, (iou, f2) in val.items():
            score = (iou + f2) / 2
            history.append(
                {
                    "epoch": epoch,
                    "arm": a,
                    "loss": float(np.mean(losses[a])),
                    "dev_iou": iou,
                    "dev_f2": f2,
                    "score": score,
                }
            )
            if score > best[a][0]:
                best[a] = (score, epoch)
                temp = OUT / f"{a}.pt.part"
                torch.save(
                    {
                        "decoder": decoders[a].state_dict(),
                        "epoch": epoch,
                        "base_decoder_sha256": models.tuned_decoder_digest(),
                        "stage": "local-mask-prompt",
                        "arm": a,
                    },
                    temp,
                )
                temp.replace(OUT / f"{a}.pt")
        print(
            f"epoch {epoch} / {EPOCHS}, dev IoU and F@2 {val}, {time.perf_counter() - start:.1f}s",
            flush=True,
        )
    with (OUT / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0])
        writer.writeheader()
        writer.writerows(history)
    assert digest(engine.model.sam_mask_decoder) == initial
    assert all(digest(getattr(engine.model, name)) == h for name, h in frozen.items())
    chosen = max(ARMS, key=lambda arm: best[arm][0])
    save_json(
        OUT / "training.json",
        {
            "seed": SEED,
            "epochs": EPOCHS,
            "train_images": len(train),
            "dev_images": len(dev),
            "selection": "mean of native dev F@2 and IoU, averaged over both prompts",
            "initial_dev_iou_f2": baseline,
            "best": best,
            "chosen": chosen,
            "encoder_digests_verified": frozen,
            "base_decoder_sha256": models.tuned_decoder_digest(),
            "lr": 2e-5,
            "optimizer": "AdamW",
            "decoder_parameters": sum(p.numel() for p in decoders[chosen].parameters()),
            "device": str(engine.device),
            "excluded_after_native_review": sorted(REJECT),
        },
    )
    print("saved", chosen, best, flush=True)


if __name__ == "__main__":
    main()
