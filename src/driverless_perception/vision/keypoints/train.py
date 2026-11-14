"""Training the keypoint network with the schedule of Dhall et al. (2019).

A config (configs/keypoints/*.yaml) has ``data`` (folder with the crop caches train.npz and val.npz), ``epochs``,
``batch``, ``lr``, ``momentum``, ``milestones`` and ``lr_factor`` (the learning rate is multiplied by it after each
milestone epoch), ``gamma`` (weight of the cross ratio term), ``time_hours`` (stop after this long), ``workers`` and
``seed``. The paper used SGD with learning rate 1e-4, momentum 0.9, batch 128, x0.1 after epochs 75 and 100 and 250
epochs in total.

The error reported is the squared distance between predicted and true keypoint in pixels of the 80x80 crop,
averaged over keypoints. The paper reached 3.78 px^2 on its test crops.
"""
from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from ...common.cone_model import cross_ratio_3d
from .crops import Augmentation, CropCache
from .network import CropDataset, KeypointNet, keypoint_loss


def evaluate_crops(model, loader, device) -> dict:
    """Mean squared keypoint error and its root, over all crops of ``loader``."""
    model.eval()
    squared = []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device))
            squared.append(((pred - y.to(device)) ** 2).sum(-1).mean(-1).cpu())
    squared = torch.cat(squared)
    return {"mse_px2": float(squared.mean()), "rmse_px": float(squared.mean().sqrt()),
            "median_px": float(squared.sqrt().median())}


def train_keypoints(config: dict, root: Path, name: str) -> Path:
    """Trains one config, returns the run folder ``experiments/keypoints/<name>``, or ``<name>-2`` and so on if
    that exists. The weights with the lowest val error are copied to ``models/keypoints/`` under the run's name,
    again never over an existing file."""
    from torch.utils.tensorboard import SummaryWriter

    from ..detection.train import free_path

    out = free_path(root / "experiments" / "keypoints" / name)
    out.mkdir(parents=True)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    data = root / config["data"]
    train_set = CropDataset(CropCache.load(data / "train.npz"), Augmentation())
    val_set = CropDataset(CropCache.load(data / "val.npz"), None)
    loader = torch.utils.data.DataLoader(train_set, config["batch"], shuffle=True, num_workers=config["workers"],
                                         worker_init_fn=CropDataset.worker_init, drop_last=True,
                                         persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_set, 256, num_workers=config["workers"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KeypointNet().to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config["milestones"], config["lr_factor"])
    # mixed precision: the full resolution layers make fp32 about three times slower
    amp = config.get("amp", True) and device == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)
    cr3d = cross_ratio_3d()
    writer = SummaryWriter(str(out))
    print(f"{len(train_set)} training crops, {len(val_set)} val crops, cross ratio of the cone {cr3d:.4f}")

    start, best = time.time(), float("inf")
    with open(out / "results.csv", "w", newline="") as log:
        table = csv.writer(log)
        table.writerow(["epoch", "time_s", "lr", "train_loss", "val_mse_px2", "val_rmse_px", "val_median_px"])
        for epoch in range(config["epochs"]):
            model.train()
            losses = []
            for x, y in loader:
                x = x.to(device, memory_format=torch.channels_last)
                with torch.autocast(device, torch.float16, enabled=amp):
                    pred = model(x)
                loss = keypoint_loss(pred.float(), y.to(device), cr3d, config["gamma"])
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
            scheduler.step()
            val = evaluate_crops(model, val_loader, device)
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            table.writerow([epoch + 1, round(elapsed, 1), lr, round(float(np.mean(losses)), 3),
                            round(val["mse_px2"], 3), round(val["rmse_px"], 3), round(val["median_px"], 3)])
            log.flush()
            writer.add_scalar("train/loss", np.mean(losses), epoch + 1)
            for k, v in val.items():
                writer.add_scalar(f"val/{k}", v, epoch + 1)
            writer.add_scalar("lr", lr, epoch + 1)
            state = {"model": model.state_dict(), "epoch": epoch + 1, "val": val, "config": config,
                     "cross_ratio": cr3d}
            torch.save(state, out / "last.pt")
            if val["mse_px2"] < best:
                best = val["mse_px2"]
                torch.save(state, out / "best.pt")
            print(f"epoch {epoch + 1}: loss {np.mean(losses):.2f}, val {val['mse_px2']:.2f} px^2 "
                  f"(median error {val['median_px']:.2f} px), {elapsed:.0f} s", flush=True)
            if elapsed > config["time_hours"] * 3600:
                print(f"stopping after {epoch + 1} epochs, time budget used up")
                break
    writer.close()
    dst = free_path(root / "models" / "keypoints" / f"{out.name}.pt")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out / "best.pt", dst)
    print(f"best weights ({best:.2f} px^2) copied to {dst}")
    return out


def load_model(path: Path, device: str = "cpu") -> KeypointNet:
    model = KeypointNet()
    model.load_state_dict(torch.load(path, map_location=device)["model"])
    return model.to(device).eval()
