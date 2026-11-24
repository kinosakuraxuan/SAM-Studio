"""Use the DAVIS J/F definitions under a box-initialized interaction protocol."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sam_app import models
from sam_app.server import Prompt
from sam_app.video import track
from training.data import DATA, SPLIT, prompt
from training.metrics import db_eval_boundary, db_eval_iou


def original_shape(packed, shape):
    return np.unpackbits(packed, count=shape[0] * shape[1]).reshape(shape).astype(bool)


def correction(truth, predicted, valid):
    """Simulate one corrective click at the center of the largest error component."""
    error = (truth != predicted) & valid
    if not np.any(error):
        return None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(error.astype(np.uint8))
    component = int(np.argmax(stats[1:n, cv2.CC_STAT_AREA])) + 1
    region = (labels == component).astype(np.uint8)
    distance = cv2.distanceTransform(np.pad(region, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(distance.argmax(), distance.shape)
    return Prompt(points=[(int(x), int(y))], labels=[int(truth[y, x])])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "eval"], default="dev")
    p.add_argument("--sequence", action="append", help="可重复，默认遍历选定划分")
    p.add_argument(
        "--version", choices=["original", "tuned", "refined"], default="original"
    )
    p.add_argument("--out", type=Path, default=models.ROOT / "outputs/video-evaluation")
    args = p.parse_args()
    names = json.loads(SPLIT.read_text())[args.split]
    selected = args.sequence or names
    if set(selected) - set(names):
        p.error("序列与划分不对应")
    args.out.mkdir(exist_ok=True, parents=True)
    model = models.make_model(args.version, video=True)
    rows = []
    for seq in selected:
        files = sorted((DATA / "Annotations/480p" / seq).glob("*.png"))
        with Image.open(files[0]) as im:
            first = np.array(im)
        frame_dir = DATA / "JPEGImages/480p" / seq
        mid = len(files) // 2
        with Image.open(files[mid]) as im:
            middle = np.array(im)
        for obj in np.setdiff1d(np.unique(first), [0, 255]):
            box = prompt(first == obj, np.random.default_rng(42), "box", 0)[2]
            keyframes = {0: Prompt(box=box)}
            original = track(frame_dir, len(files), keyframes, model)
            click = correction(
                middle == obj, original_shape(original[mid], first.shape), middle != 255
            )
            if click is not None:
                keyframes[mid] = click
                corrected = track(frame_dir, len(files), keyframes, model)
            else:
                corrected = original
            j_values, f_values, after_before, after_new = [], [], [], []
            for i in range(1, len(files)):
                with Image.open(files[i]) as im:
                    ids = np.array(im)
                pred = original_shape(original[i], ids.shape)
                ref, void = ids == obj, ids == 255
                j = float(db_eval_iou(ref, pred, void))
                f = float(db_eval_boundary(ref, pred, void))
                j_values.append(j)
                f_values.append(f)
                if i > mid:
                    after_before.append(j)
                    after_new.append(
                        float(
                            db_eval_iou(
                                ref, original_shape(corrected[i], ids.shape), void
                            )
                        )
                    )
            j, f = float(np.mean(j_values)), float(np.mean(f_values))
            rows.append(
                [
                    seq,
                    int(obj),
                    len(files),
                    j,
                    f,
                    (j + f) / 2,
                    mid,
                    int(click is not None),
                    float(np.mean(after_before)),
                    float(np.mean(after_new)),
                ]
            )
    stem = (
        args.split
        + "-"
        + args.version
        + ("-" + "-".join(selected) if args.sequence else "")
    )
    with (args.out / (stem + ".csv")).open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(
            [
                "video",
                "object",
                "frames",
                "J",
                "F",
                "J_and_F",
                "correction_frame",
                "clicks",
                "subsequent_J_before",
                "subsequent_J_after",
            ]
        )
        writer.writerows(rows)
    print(
        "Sequences:",
        len(selected),
        "objects:",
        len(rows),
        "mean J&F:",
        float(np.mean([r[5] for r in rows])),
    )


if __name__ == "__main__":
    main()
