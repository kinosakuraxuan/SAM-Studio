"""A residual boundary branch for SAM 2 mask logits."""

import torch
from torch import nn
from torch.nn import functional as F


class BoundaryRefiner(nn.Module):
    def __init__(self, feature_channels):
        super().__init__()
        self.project = nn.Conv2d(feature_channels, 8, 1)
        self.correction = nn.Sequential(
            nn.Conv2d(9, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, logits, high_res_feature):
        batch, masks = logits.shape[:2]
        features = self.project(high_res_feature).expand(batch, -1, -1, -1)
        features = features[:, None].expand(-1, masks, -1, -1, -1)
        confidence = torch.tanh(logits / 4).unsqueeze(2)
        inputs = torch.cat((features, confidence), dim=2)
        residual = self.correction(inputs.flatten(0, 1)).reshape_as(logits)

        # Limit corrections to three decoder pixels on either side of a contour.
        foreground = (logits > 0).to(residual.dtype)
        outer = F.max_pool2d(foreground, 7, stride=1, padding=3)
        inner = 1 - F.max_pool2d(1 - foreground, 7, stride=1, padding=3)
        return logits + (residual * (outer - inner)).to(logits.dtype)

    def refine_decoder_output(self, decoder, args, kwargs, output):
        logits, quality, tokens, object_scores = output
        fine_feature = kwargs["high_res_features"][0]
        return self(logits, fine_feature), quality, tokens, object_scores


def attach_boundary_refiner(model):
    decoder = model.sam_mask_decoder
    projection = decoder.conv_s0
    refiner = BoundaryRefiner(projection.out_channels).to(
        device=projection.weight.device, dtype=projection.weight.dtype
    )
    model.boundary_refiner = refiner
    decoder.register_forward_hook(refiner.refine_decoder_output, with_kwargs=True)
    refiner.train(model.training)
    return refiner
