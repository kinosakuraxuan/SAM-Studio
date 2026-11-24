"""Run from the project root: python -m training.train_mask_decoder"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sam_app.models import REFINED, ROOT, device, make_model, tuned_decoder_digest
from sam_app.refinement import attach_boundary_refiner
from training.data import DATA, SPLIT, prompt, read_target, samples, scores
from training.metrics import db_eval_boundary


def predict_logits(model, transforms, rgb, points, labels, box):
    # The image encoder has no trainable parameters. The high-resolution
    # decoder projections inside forward_image retain their gradient path.
    tensor = transforms(rgb).unsqueeze(0).to(device())
    output = model.forward_image(tensor)
    _, flat, _, sizes = model._prepare_backbone_features(output)
    if model.directly_add_no_mem_embed:
        flat[-1] = flat[-1] + model.no_mem_embed
    features = [f.permute(1, 2, 0).reshape(1, -1, *s) for f, s in zip(flat, sizes)]
    if box is not None:
        points = [[box[0], box[1]], [box[2], box[3]]] + points
        labels = [2, 3] + labels
    xy = torch.tensor(points, device=device(), dtype=torch.float32).unsqueeze(0)
    xy = transforms.transform_coords(xy, normalize=True, orig_hw=rgb.shape[:2])
    lab = torch.tensor(labels, device=device(), dtype=torch.int32).unsqueeze(0)
    sparse, dense = model.sam_prompt_encoder(points=(xy, lab), boxes=None, masks=None)
    masks, _, _, _ = model.sam_mask_decoder(
        image_embeddings=features[-1],
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        repeat_image=False,
        high_res_features=features[:-1],
    )
    return F.interpolate(
        masks.float(), size=rgb.shape[:2], mode="bilinear", align_corners=False
    )[0, 0]


def loss_fn(logits, target, valid):
    keep = torch.from_numpy(valid).to(logits.device)
    truth = torch.from_numpy(target).to(logits.device, torch.float32)[keep]
    pred = logits[keep]
    prob = pred.sigmoid()
    dice = (2 * (prob * truth).sum() + 1) / (prob.sum() + truth.sum() + 1)
    return F.binary_cross_entropy_with_logits(pred, truth) + 1 - dice


@torch.no_grad()
def evaluate(model, transforms, pairs, seed, measure_boundary=False):
    model.eval()
    rng = np.random.default_rng(seed)
    vals = []
    boundaries = []
    for pair in pairs:
        item = read_target(pair, rng)
        if item is None:
            continue
        rgb, target, valid, _ = item
        for mode in ("point", "box"):
            p = prompt(target, rng, mode)
            logits = predict_logits(model, transforms, rgb, *p)
            mask = logits.cpu().numpy() > 0
            vals.append(scores(mask, target, valid)[0])
            if measure_boundary:
                boundaries.append(db_eval_boundary(target, mask, ~valid))
    if not vals:
        raise ValueError("开发集没有可见实例。")
    return float(np.mean(vals)), float(
        np.mean(boundaries)
    ) if measure_boundary else None


def main():
    from sam2.utils.transforms import SAM2Transforms

    parser = argparse.ArgumentParser(
        description="Fine-tune SAM 2.1 mask decoder on DAVIS"
    )
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--per-video", type=int, default=10)
    parser.add_argument("--limit", type=int, help="训练与开发冒烟验证的帧数上限")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--refine",
        action="store_true",
        help="冻结已微调的 SAM，仅训练新增的卷积边界细化分支",
    )
    args = parser.parse_args()
    if min(args.epochs, args.per_video, args.limit or 1) < 1:
        parser.error("轮次和样本数必须为正数")
    if args.out is None:
        args.out = REFINED.parent if args.refine else ROOT / "outputs/training"
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "history.csv").exists():
        parser.error("该输出目录已有训练记录，请为新实验选择 --out")
    split = json.loads(SPLIT.read_text())
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train = samples(args.data, split["train"], args.per_video)
    dev = samples(args.data, split["dev"], args.per_video)
    if args.limit:
        train = train[: args.limit]
        dev = dev[: args.limit]
    config = {
        **vars(args),
        "train_frames": len(train),
        "dev_frames": len(dev),
        "lr": 0.0001,
        "accumulation": 4,
        "split": split,
        "torch": str(torch.__version__),
    }
    base_digest = tuned_decoder_digest() if args.refine else None
    model = make_model("tuned" if args.refine else "original")
    trainable = (
        attach_boundary_refiner(model) if args.refine else model.sam_mask_decoder
    )
    model.requires_grad_(False)
    trainable.requires_grad_(True)
    if args.refine:
        if tuned_decoder_digest() != base_digest:
            raise RuntimeError("加载时解码器权重发生变化，请重试。")
        config["base_decoder_sha256"] = base_digest
        config["trainable_parameters"] = sum(p.numel() for p in trainable.parameters())
    (args.out / "config.json").write_text(json.dumps(config, default=str, indent=2))
    optimizer = torch.optim.AdamW(trainable.parameters(), lr=0.0001)
    transforms = SAM2Transforms(model.image_size, mask_threshold=0)
    best = -1
    with (args.out / "history.csv").open("x", newline="") as log:
        writer = csv.writer(log)
        columns = ["epoch", "loss", "dev_iou", "grad_norm", "seconds"]
        if args.refine:
            columns.append("dev_boundary_f")
        writer.writerow(columns)
        for epoch in range(args.epochs):
            started = time.perf_counter()
            trainable.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = seen = pending = 0
            order = rng.permutation(len(train))
            for index in order:
                item = read_target(train[index], rng)
                if item is None:
                    continue
                rgb, truth, valid, _ = item
                mode = "point" if rng.random() < 0.5 else "box"
                logits = predict_logits(
                    model, transforms, rgb, *prompt(truth, rng, mode)
                )
                loss = loss_fn(logits, truth, valid)
                loss.backward()
                seen += 1
                pending += 1
                loss_sum += loss.item()
                if pending == 4:
                    grad = step(trainable, optimizer, pending)
                    pending = 0
            if not seen:
                raise ValueError("训练样本均无可见目标。")
            if pending:
                grad = step(trainable, optimizer, pending)
            iou, boundary = evaluate(model, transforms, dev, args.seed, args.refine)
            row = [
                epoch + 1,
                loss_sum / seen,
                iou,
                grad,
                time.perf_counter() - started,
            ]
            if args.refine:
                row.append(boundary)
            writer.writerow(row)
            log.flush()
            print(dict(zip(columns, row)), flush=True)
            metric = boundary if args.refine else iou
            if metric > best:
                best = metric
                tmp = args.out / "best.pt.part"
                weights = {"epoch": epoch + 1, "dev_iou": iou}
                if args.refine:
                    weights.update(
                        refiner=trainable.state_dict(),
                        dev_boundary_f=boundary,
                        base_decoder_sha256=base_digest,
                    )
                else:
                    weights["decoder"] = trainable.state_dict()
                torch.save(weights, tmp)
                tmp.replace(args.out / "best.pt")


def step(decoder, optimizer, n):
    for param in decoder.parameters():
        if param.grad is not None:
            param.grad.div_(n)
    norm = float(
        torch.nn.utils.clip_grad_norm_(
            decoder.parameters(), 1.0, error_if_nonfinite=True
        )
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return norm


if __name__ == "__main__":
    main()
