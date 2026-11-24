"""Refine a prompt-selected mask on the original image grid."""

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def conv_block(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(outputs, outputs, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class PixelRefiner(nn.Module):
    def __init__(self, use_rgb=True):
        super().__init__()
        self.use_rgb = use_rgb
        self.fine = conv_block(4, 12)
        self.middle = conv_block(12, 24)
        self.low = conv_block(24, 32)
        self.up_middle = conv_block(56, 24)
        self.up_fine = conv_block(36, 12)
        self.residual = nn.Conv2d(12, 1, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, rgb, coarse):
        detail = rgb / 127.5 - 1 if self.use_rgb else torch.zeros_like(rgb)
        fine = self.fine(torch.cat((detail, coarse.clamp(-16, 16) / 8), dim=1))
        middle = self.middle(F.max_pool2d(fine, 2, ceil_mode=True))
        low = self.low(F.max_pool2d(middle, 2, ceil_mode=True))
        up = self.up_middle(torch.cat((middle, upsample(low, middle)), dim=1))
        result = self.up_fine(torch.cat((fine, upsample(up, fine)), dim=1))
        return coarse + self.residual(result)


def upsample(source, target):
    return F.interpolate(
        source, size=target.shape[-2:], mode="bilinear", align_corners=False
    )


def refinement_bounds(logits, points=(), labels=(), box=None):
    h, w = logits.shape
    x, y, mw, mh = cv2.boundingRect((logits > 0).astype(np.uint8))
    xs = [a for (a, _), label in zip(points, labels) if label == 1]
    ys = [b for (_, b), label in zip(points, labels) if label == 1]
    if mw and mh:
        xs.extend((x, x + mw))
        ys.extend((y, y + mh))
    if box is not None:
        xs.extend((box[0], box[2]))
        ys.extend((box[1], box[3]))
    if not xs:
        return None
    margin = max(24, int(0.08 * max(max(xs) - min(xs), max(ys) - min(ys))))
    return (
        max(0, int(min(xs)) - margin),
        max(0, int(min(ys)) - margin),
        min(w, int(np.ceil(max(xs))) + margin),
        min(h, int(np.ceil(max(ys))) + margin),
    )


@torch.inference_mode()
def refine_pixels(model, rgb, coarse, bounds):
    if bounds is None:
        return coarse
    x0, y0, x1, y1 = bounds
    device = next(model.parameters()).device
    # Disjoint output tiles have a halo so convolutions see nearby context.
    result = coarse.copy()
    side, halo = 320, 32
    for y in range(y0, y1, side):
        for x in range(x0, x1, side):
            yb, xb = min(y + side, y1), min(x + side, x1)
            a, b = max(y - halo, 0) // 4 * 4, max(x - halo, 0) // 4 * 4
            c = min(((yb + halo + 3) // 4) * 4, coarse.shape[0])
            d = min(((xb + halo + 3) // 4) * 4, coarse.shape[1])
            pixels = (
                torch.from_numpy(rgb[a:c, b:d].copy()).permute(2, 0, 1)[None].to(device)
            )
            logits = torch.from_numpy(coarse[a:c, b:d].copy())[None, None].to(device)
            padding = (0, (b - d) % 4, 0, (a - c) % 4)
            if any(padding):
                pixels = F.pad(pixels, padding, mode="replicate")
                logits = F.pad(logits, padding, mode="replicate")
            refined = model(pixels, logits)
            result[y:yb, x:xb] = (
                refined[0, 0, y - a : yb - a, x - b : xb - b].cpu().numpy()
            )
    return result
