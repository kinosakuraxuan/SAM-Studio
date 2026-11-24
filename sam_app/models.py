import hashlib
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models/sam2.1_hiera_tiny.pt"
TUNED = ROOT / "outputs/training/best.pt"
REFINED = ROOT / "outputs/refinement-training/best.pt"
DINO = ROOT / "models/grounding-dino-tiny"
CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SOURCE = ROOT / "vendor/sam2"
if SOURCE.is_dir():
    sys.path.append(str(SOURCE))

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def tuned_decoder_digest():
    with TUNED.open("rb") as weights:
        return hashlib.file_digest(weights, "sha256").hexdigest()


def make_model(version="original", video=False, *, refiner_checkpoint=None):
    if not CHECKPOINT.is_file():
        raise FileNotFoundError("缺少 SAM 权重，请先运行资源准备脚本。")
    build = build_sam2_video_predictor if video else build_sam2
    model = build(CONFIG, str(CHECKPOINT), device=device(), apply_postprocessing=False)
    refinement = None
    if version == "refined":
        path = refiner_checkpoint or REFINED
        if not path.is_file():
            raise FileNotFoundError("缺少本地边界细化权重。")
        refinement = torch.load(path, map_location="cpu", weights_only=True)
    if version in ("tuned", "refined"):
        if not TUNED.is_file():
            raise FileNotFoundError("请先训练并保存解码器权重。")
        with TUNED.open("rb") as file:
            if refinement is not None:
                digest = hashlib.file_digest(file, "sha256").hexdigest()
                if digest != refinement["base_decoder_sha256"]:
                    raise ValueError("细化权重与当前解码器不匹配，请重新训练细化分支。")
                file.seek(0)
            weights = torch.load(file, map_location="cpu", weights_only=True)
        model.sam_mask_decoder.load_state_dict(weights["decoder"])
    elif version != "original":
        raise ValueError("未知权重版本")
    if refinement is not None:
        from .refinement import attach_boundary_refiner

        attach_boundary_refiner(model).load_state_dict(refinement["refiner"])
    return model


def predictor(version="original", *, refiner_checkpoint=None):
    return SAM2ImagePredictor(
        make_model(version, refiner_checkpoint=refiner_checkpoint)
    )


def mask_from_prompt(engine, points, labels, box):
    return prompt_prediction(engine, points, labels, box, return_logits=False).astype(
        bool
    )


def prompt_prediction(engine, points, labels, box, *, return_logits=True):
    if not points and box is None:
        raise ValueError("请先添加一个点或拖出目标框。")
    masks, _, _ = engine.predict(
        point_coords=np.asarray(points, dtype=np.float32) if points else None,
        point_labels=np.asarray(labels, dtype=np.int32) if points else None,
        box=np.asarray(box, dtype=np.float32) if box is not None else None,
        multimask_output=False,
        return_logits=return_logits,
    )
    return masks[0]


WORDS = {
    "人": "person",
    "自行车": "bicycle",
    "汽车": "car",
    "狗": "dog",
    "猫": "cat",
    "背包": "backpack",
    "红色背包": "red backpack",
    "卡车": "truck",
    "公交车": "bus",
    "摩托车": "motorcycle",
    "马": "horse",
    "船": "boat",
    "滑板": "skateboard",
}


@lru_cache(maxsize=1)
def detector():
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    if not (DINO / "model.safetensors").is_file():
        raise FileNotFoundError("缺少本地 Grounding DINO Tiny 权重。")
    processor = AutoProcessor.from_pretrained(DINO, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        DINO, local_files_only=True, use_safetensors=True
    ).eval()
    return processor, model


@torch.inference_mode()
def candidates(rgb, phrase, threshold=0.30):
    phrase = phrase.strip().lower().rstrip(".。").strip()
    phrase = WORDS.get(phrase, phrase)
    if not phrase or not phrase.isascii():
        raise ValueError("请输入英文描述，或使用已收录的中文词，如“自行车”。")
    processor, model = detector()
    # Keep the detector on CPU so video tracking can retain its GPU state.
    inputs = processor(
        images=Image.fromarray(rgb), text=phrase + ".", return_tensors="pt"
    )
    output = processor.post_process_grounded_object_detection(
        model(**inputs),
        inputs.input_ids,
        threshold=threshold,
        text_threshold=0.25,
        target_sizes=[rgb.shape[:2]],
    )[0]
    h, w = rgb.shape[:2]
    boxes = []
    for box, score in zip(output["boxes"], output["scores"]):
        x, y, right, bottom = np.clip(box.tolist(), 0, [w, h, w, h]).tolist()
        if right > x and bottom > y:
            boxes.append({"box": [x, y, right, bottom], "score": float(score)})
    return sorted(boxes, key=lambda row: row["score"], reverse=True)
