"""Deterministic foreground rendering on decoded RGB pixels."""

from typing import Literal, get_args

import cv2
import numpy as np

Style = Literal["neural-cartoon", "neural-sketch"]


def stylize(
    rgb: np.ndarray, mask: np.ndarray, style: Style, strength: int = 100, window=None
) -> np.ndarray:
    if style not in get_args(Style) or not 0 <= strength <= 100:
        raise ValueError("风格或强度超出范围")
    if rgb.dtype != np.uint8 or rgb.shape != (*mask.shape, 3):
        raise ValueError("RGB 图像与掩膜尺寸不匹配")
    result = rgb.copy()
    if strength == 0 or not mask.any():
        return result
    x, y, w, h = window or cv2.boundingRect(mask.astype(np.uint8))
    ys = slice(max(0, y - 18), min(rgb.shape[0], y + h + 18))
    xs = slice(max(0, x - 18), min(rgb.shape[1], x + w + 18))
    roi = np.ascontiguousarray(rgb[ys, xs])
    from .neural_styles import render

    effect = render(roi, style)
    weight = strength / 100.0
    blended = cv2.addWeighted(roi, 1 - weight, effect, weight, 0)
    np.copyto(result[ys, xs], blended, where=mask[ys, xs, None].astype(bool))
    return result
