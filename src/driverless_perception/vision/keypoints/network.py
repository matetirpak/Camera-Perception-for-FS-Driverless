"""The keypoint network of Dhall et al. (2019) and its loss.

An 80x80 crop, a convolution and four residual blocks with 64, 128, 256 and 512 channels, then a fully connected
layer with 14 outputs: the keypoints in pixels of the crop. The loss is the squared keypoint error plus gamma
times the squared cross ratio error of both edges.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .crops import CROP, Augmentation, CropCache, augmented_crop, plain_crop
from .cross_ratio import LEFT_EDGE, RIGHT_EDGE


class ResidualBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout),
        )
        self.skip = nn.Identity() if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return torch.relu(self.body(x) + self.skip(x))


class KeypointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResidualBlock(64, 64, 1), ResidualBlock(64, 128, 2), ResidualBlock(128, 256, 2),
            ResidualBlock(256, 512, 2), nn.AdaptiveAvgPool2d(5),
        )
        self.head = nn.Linear(512 * 5 * 5, 14)

    def forward(self, x):
        """(B, 3, 80, 80) RGB in [0, 1] to (B, 7, 2) keypoints in crop pixels."""
        return self.head(self.features(x).flatten(1)).view(-1, 7, 2) + CROP / 2


def cross_ratio(p: torch.Tensor, idx) -> torch.Tensor:
    """Cross ratio of four points (..., 7, 2) picked by ``idx``."""
    a, b, c, d = (p[..., i, :] for i in idx)
    dist = lambda u, v: torch.linalg.norm(u - v, dim=-1).clamp_min(1e-6)
    return (dist(a, c) / dist(a, d)) / (dist(b, c) / dist(b, d))


def keypoint_loss(pred: torch.Tensor, target: torch.Tensor, cr3d: float, gamma: float = 1e-4) -> torch.Tensor:
    squared = ((pred - target) ** 2).sum(dim=(-1, -2))
    cr = (cross_ratio(pred, LEFT_EDGE) - cr3d) ** 2 + (cross_ratio(pred, RIGHT_EDGE) - cr3d) ** 2
    return (squared + gamma * cr).mean()


def to_tensor(crops_bgr: np.ndarray) -> torch.Tensor:
    """(B, 80, 80, 3) or (80, 80, 3) uint8 BGR to float RGB tensors in [0, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(crops_bgr[..., ::-1])).float() / 255
    return x.permute(0, 3, 1, 2) if x.dim() == 4 else x.permute(2, 0, 1)


class CropDataset(torch.utils.data.Dataset):
    """Jittered crops for training, plain box crops for evaluation."""

    def __init__(self, cache: CropCache, augment: Augmentation | None):
        self.cache, self.augment = cache, augment
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        image, box, kp = self.cache.image(i), self.cache.boxes[i], self.cache.keypoints[i]
        if self.augment is None:
            crop, target, _ = plain_crop(image, box, kp)
        else:
            crop, target = augmented_crop(image, box, kp, self.rng, self.augment)
        return to_tensor(crop), torch.from_numpy(target.astype(np.float32))

    @staticmethod
    def worker_init(worker_id: int) -> None:
        info = torch.utils.data.get_worker_info()
        info.dataset.rng = np.random.default_rng(info.seed % 2**32)
