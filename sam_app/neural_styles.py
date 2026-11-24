"""Local inference for White-box Cartoonization and Informative Drawings."""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

WEIGHTS = Path(__file__).resolve().parents[1] / "models"


def activated(src, dst, kernel=3, stride=1):
    return [nn.Conv2d(src, dst, kernel, stride, kernel // 2), nn.LeakyReLU(0.2)]


class CartoonResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            *activated(128, 128), nn.Conv2d(128, 128, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class Cartoon(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_down = nn.Sequential(*activated(3, 32, 7))
        self.down1 = nn.Sequential(*activated(32, 32, stride=2), *activated(32, 64))
        self.down2 = nn.Sequential(*activated(64, 64, stride=2), *activated(64, 128))
        self.res_blocks = nn.Sequential(*(CartoonResidual() for _ in range(4)))
        self.up1 = nn.Sequential(*activated(128, 64))
        self.up2 = nn.Sequential(*activated(64, 64), *activated(64, 32))
        self.last = nn.Sequential(*activated(32, 32), nn.Conv2d(32, 3, 7, padding=3))

    def forward(self, x):
        a = self.initial_down(x * 2 - 1)
        b = self.down1(a)
        c = self.up1(self.res_blocks(self.down2(b)))
        c = self.up2(
            F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False)
            + b
        )
        out = self.last(
            F.interpolate(c, size=a.shape[-2:], mode="bilinear", align_corners=False)
            + a
        )
        return (out.tanh() + 1) / 2


class InkResidual(nn.Module):
    def __init__(self):
        super().__init__()

        def layers():
            return [
                nn.ReflectionPad2d(1),
                nn.Conv2d(256, 256, 3),
                nn.InstanceNorm2d(256),
            ]

        self.conv_block = nn.Sequential(*layers(), nn.ReLU(), *layers())

    def forward(self, x):
        return x + self.conv_block(x)


class Ink(nn.Module):
    def __init__(self):
        super().__init__()
        self.model0 = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(3, 64, 7), nn.InstanceNorm2d(64), nn.ReLU()
        )
        down, up = [], []
        for n in (64, 128):
            down.extend(
                [nn.Conv2d(n, n * 2, 3, 2, 1), nn.InstanceNorm2d(n * 2), nn.ReLU()]
            )
        for n in (256, 128):
            up.extend(
                [
                    nn.ConvTranspose2d(n, n // 2, 3, 2, 1, output_padding=1),
                    nn.InstanceNorm2d(n // 2),
                    nn.ReLU(),
                ]
            )
        self.model1 = nn.Sequential(*down)
        self.model2 = nn.Sequential(*(InkResidual() for _ in range(3)))
        self.model3 = nn.Sequential(*up)
        self.model4 = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(64, 1, 7), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model4(
            self.model3(self.model2(self.model1(self.model0(x))))
        ).expand(-1, 3, -1, -1)


@lru_cache(maxsize=2)
def network(style):
    cartoon = style == "neural-cartoon"
    path = WEIGHTS / ("whitebox.pth" if cartoon else "lineart.pth")
    if not path.is_file():
        raise FileNotFoundError(f"缺少本地风格权重 {path.name}")
    model = Cartoon() if cartoon else Ink()
    state = torch.load(path, weights_only=True, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state))
    return model.eval().to("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def render(rgb, style):
    model = network(style)
    h, w = rgb.shape[:2]
    scale = min(1.0, 512 / max(w, h))
    resized = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))))
    rh, rw = resized.shape[:2]
    padded = cv2.copyMakeBorder(
        resized,
        0,
        max(16, (rh + 3) // 4 * 4) - rh,
        0,
        max(16, (rw + 3) // 4 * 4) - rw,
        cv2.BORDER_REPLICATE,
    )
    tensor = (
        torch.from_numpy(padded)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(next(model.parameters()).device)
        .float()
        / 255
    )
    output = (
        model(tensor)[0, :, :rh, :rw]
        .permute(1, 2, 0)
        .mul(255)
        .round()
        .clamp(0, 255)
        .byte()
        .cpu()
        .numpy()
    )
    if style == "neural-cartoon":
        # Guided filtering with the input image suppresses generator texture at flat surfaces.
        guide = resized.astype(np.float32) / 255
        raw = output.astype(np.float32) / 255
        mean = lambda x: cv2.blur(x, (3, 3))
        g, y = mean(guide), mean(raw)
        a = (mean(guide * raw) - g * y) / (mean(guide * guide) - g * g + 0.005)
        output = (
            ((mean(a) * guide + mean(y - a * g)).clip(0, 1) * 255)
            .round()
            .astype(np.uint8)
        )
    return cv2.resize(output, (w, h))
