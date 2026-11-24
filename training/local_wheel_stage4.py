"""Controlled ablation of the final Hiera stage in the local full-resolution pass."""

import argparse
import csv
import hashlib
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
from training.evaluate_local_wheels import measurements
from training.train_local_wheels import (
    EPOCHS,
    SEED,
    backbone,
    decode,
    digest,
    objective,
    prepare,
    validation,
)
from training.wheel_samples import OUT as CONTROL

OUT = models.ROOT / "outputs/local-encoder-stage4"
# This region brackets only the third spoke on each of 21 consecutive annotation rows.
B3_RECT = (226, 455, 240, 476)


def last_stage(engine):
    trunk = engine.model.image_encoder.trunk
    indices = list(range(trunk.stage_ends[-2] + 1, trunk.stage_ends[-1] + 1))
    return indices, trunk.blocks[indices[0] : indices[-1] + 1]


def frozen_digest(model):
    h = hashlib.sha256()
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    for name, t in model.state_dict().items():
        if name not in trainable:
            h.update(name.encode())
            h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def control_checkpoint():
    info = json.loads((CONTROL / "training.json").read_text())
    assert (
        info["chosen"] == "weighted"
        and info["base_decoder_sha256"] == models.tuned_decoder_digest()
    )
    return torch.load(CONTROL / "weighted.pt", map_location="cpu", weights_only=True)


def train():
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    engine = models.predictor("tuned")
    model = engine.model
    model.eval().requires_grad_(False)
    samples = prepare(engine, OUT / "selected.json")
    if json.loads((OUT / "selected.json").read_text()) != json.loads(
        (CONTROL / "selected.json").read_text()
    ):
        raise ValueError("Original image, split, crop or prompts changed")
    training = [s for s in samples if s[0].startswith("tr")]
    dev = [s for s in samples if s[0].startswith("vd")]
    del samples
    encoder_indices, stage = last_stage(engine)
    decoder = model.sam_mask_decoder
    initial_decoder = digest(decoder)
    initial_blocks = [digest(block) for block in stage]
    # Re-evaluate the actual stage-3 checkpoint on these exact validation inputs.
    initial_state = {
        k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()
    }
    ctrl = control_checkpoint()
    decoder.load_state_dict(ctrl["decoder"])
    control_dev = validation(engine, {"decoder": decoder}, dev)["decoder"]
    with (CONTROL / "history.csv").open() as f:
        archived = next(
            r
            for r in csv.DictReader(f)
            if r["arm"] == "weighted" and int(r["epoch"]) == ctrl["epoch"]
        )
    np.testing.assert_allclose(
        control_dev,
        [float(archived["dev_iou"]), float(archived["dev_f2"])],
        atol=1e-6,
        rtol=0,
    )
    decoder.load_state_dict(initial_state)
    assert digest(decoder) == initial_decoder
    stage.requires_grad_(True)
    decoder.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    num_stage = sum(p.numel() for p in stage.parameters())
    num_decoder = sum(p.numel() for p in decoder.parameters())
    assert sum(p.numel() for p in params) == num_stage + num_decoder
    initial_frozen = frozen_digest(model)
    optimizer = torch.optim.AdamW(params, lr=2e-5, weight_decay=0.01)
    best = -1
    history = []
    for epoch in range(1, EPOCHS + 1):
        stage.train()
        decoder.train()
        losses = []
        for idx in rng.permutation(len(training)):
            _, rgb, gt, priors, weight = training[idx]
            optimizer.zero_grad(set_to_none=True)
            feats = backbone(engine, rgb, train_encoder=True)
            target = torch.from_numpy(gt).to(engine.device, torch.float32)
            weights = torch.from_numpy(weight).to(engine.device)
            for i, prior in enumerate(priors):
                pred = decode(engine, decoder, feats, prior, gt.shape)
                loss = objective(pred, target, weights)
                (loss / len(priors)).backward(retain_graph=i < len(priors) - 1)
                losses.append(float(loss.detach()))
            if not all(
                any(p.grad is not None for p in block.parameters()) for block in stage
            ):
                raise RuntimeError("Disconnected encoder stage")
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
        stage.eval()
        iou, f2 = validation(engine, {"joint": decoder}, dev)["joint"]
        score = (iou + f2) / 2
        history.append(
            {
                "epoch": epoch,
                "loss": np.mean(losses),
                "iou": iou,
                "f2": f2,
                "score": score,
            }
        )
        if score > best:
            best = score
            temp = OUT / "joint.pt.part"
            torch.save(
                {
                    "decoder": decoder.state_dict(),
                    "stage": stage.state_dict(),
                    "indices": encoder_indices,
                    "epoch": epoch,
                    "base_decoder_sha256": models.tuned_decoder_digest(),
                },
                temp,
            )
            temp.replace(OUT / "joint.pt")
        print(f"epoch {epoch}  dev IoU {iou:.4f}  F@2 {f2:.4f}", flush=True)
    assert initial_frozen == frozen_digest(model)
    saved = torch.load(OUT / "joint.pt", map_location="cpu", weights_only=True)
    stage.load_state_dict(saved["stage"])
    decoder.load_state_dict(saved["decoder"])
    assert all(digest(b) != old for b, old in zip(stage, initial_blocks, strict=True))
    assert digest(decoder) != initial_decoder
    with (OUT / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0])
        writer.writeheader()
        writer.writerows(history)
    save_json(
        OUT / "training.json",
        {
            "indices": encoder_indices,
            "stage_parameters": num_stage,
            "decoder_parameters": num_decoder,
            "train_images": len(training),
            "dev_images": len(dev),
            "seed": SEED,
            "epochs": EPOCHS,
            "lr": 2e-5,
            "control_epoch": ctrl["epoch"],
            "control_dev_iou_f2": control_dev,
            "joint_epoch": saved["epoch"],
            "joint_dev": history[saved["epoch"] - 1],
            "frozen_state_sha256_verified": initial_frozen,
            "initial_decoder_digest": initial_decoder,
            "base_decoder_sha256": models.tuned_decoder_digest(),
        },
    )


@torch.inference_mode()
def evaluate():
    previous = json.loads((CONTROL / "wheel-scores.json").read_text())
    with ZipFile(ARCHIVE) as archive:
        _, record, roi, _ = next(sample_set())
        rgb, _, _, target, valid, _ = frame_from_archive(
            archive, record, "Full-Resolution"
        )
    assert list(roi) == previous["roi"]
    context = region(previous["context"])
    field = region(roi)
    x, y, r, b = roi
    a, c, _, _ = previous["context"]
    inner = region((x - a, y - c, r - a, b - c))
    engine = models.predictor("tuned")
    engine.set_image(rgb)
    priors = {
        mode: mask_prompt(models.prompt_prediction(engine, *cue)[context])
        for mode, cue in previous["native_prompts"].items()
    }
    gt = target[field]
    good = valid[field]
    preds = {}
    scores = {}
    with np.load(CONTROL / "wheel-predictions.npz") as old:
        assert (
            np.array_equal(gt, old["truth"])
            and np.array_equal(good, old["valid"])
            and np.array_equal(rgb[field], old["rgb"])
        )
        zones = {"wheel": np.ones_like(gt), "tire": old["tire"], "frame": old["frame"]}
        for code, (x, y, r, b, _) in DETAILS.items():
            zone = np.zeros_like(gt)
            zone[y:b, x:r] = True
            zones[code] = zone
        zone = np.zeros_like(gt)
        strip = region(B3_RECT)
        zone[strip] = True
        zones["B3"] = zone
        stripe = gt[strip]
        edges = np.diff(np.pad(stripe.astype(np.int8), ((0, 0), (1, 1))), axis=1)
        assert ((edges == 1).sum(axis=1) == 1).all() and not stripe[:, [0, -1]].any()
        for arm in ("decoder", "joint"):
            if arm == "decoder":
                state = control_checkpoint()
            else:
                state = torch.load(
                    OUT / "joint.pt", map_location="cpu", weights_only=True
                )
                indices, stage = last_stage(engine)
                assert (
                    indices == state["indices"]
                    and state["base_decoder_sha256"] == models.tuned_decoder_digest()
                )
                stage.load_state_dict(state["stage"])
            engine.model.sam_mask_decoder.load_state_dict(state["decoder"])
            engine.set_image(np.ascontiguousarray(rgb[context]))
            for mode, prior in priors.items():
                mask = (
                    engine.predict(
                        mask_input=prior, multimask_output=False, return_logits=True
                    )[0][0]
                    > 0
                )
                cropped = mask[inner].copy()
                baseline = old[f"{mode}_weighted"]
                if arm == "decoder" and not np.array_equal(cropped, baseline):
                    raise RuntimeError(
                        "Previous wheel control differs at one or more pixels"
                    )
                key = f"{mode}_{arm}"
                preds[key] = cropped
                scores[key] = {
                    k: measurements(gt, good & v, baseline, cropped)
                    for k, v in zones.items()
                }
                scores[key]["B3"]["rows_hit"] = int(
                    (stripe & cropped[strip]).any(axis=1).sum()
                )
                scores[key]["wheel"]["f2"] = wheel_metrics(
                    target[context], valid[context], np.zeros_like(mask), mask, inner
                )["local_f2"]
                scores[key]["wheel"]["corrected_background"] = int(
                    (good & ~gt & baseline & ~cropped).sum()
                )
                scores[key]["wheel"]["new_background_errors"] = int(
                    (good & ~gt & ~baseline & cropped).sum()
                )
    np.savez_compressed(
        OUT / "wheel.npz", rgb=rgb[field], truth=gt, valid=good, **preds
    )
    cut = spoke_audit(gt, good, preds)
    assert cut["B"]["foreground_runs"] == previous["transects"]["B"]["foreground_runs"]
    save_json(
        OUT / "wheel.json",
        {
            "case": previous["case"],
            "roi": roi,
            "context": previous["context"],
            "prompts": previous["native_prompts"],
            "verified_step3_predictions": True,
            "B3_rect": B3_RECT,
            "scores": scores,
            "transects": cut,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "evaluate"))
    cv2.setNumThreads(2)
    if parser.parse_args().action == "train":
        train()
    else:
        evaluate()
