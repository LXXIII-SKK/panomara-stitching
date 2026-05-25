#!/usr/bin/env python
"""
PhamHungSon_15_CNN_Train_Sup.py

Clean supervised training script for 4-point homography regression.

Expected dataset structure:

data/cnn/split/
    Train_synthetic/
        PA/
            000001.png
            ...
        PB/
            000001.png
            ...
        H4.csv
    Val_synthetic/
        PA/
        PB/
        H4.csv

H4.csv format:
    filename, dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4

This script:
    - Uses explicit Dataset/DataLoader instead of random batch generation.
    - Reads images as grayscale explicitly.
    - Normalizes input images to [-1, 1].
    - Trains on normalized labels: label_px / label_scale.
    - Reports metrics in real pixel units.
    - Supports ConvNet and ResNet18 backbones.
    - Supports pair-difference/coordinate input channels for better translation sensitivity.
    - Supports a corner-aware loss term that directly optimizes MCE.
    - Supports optional CUDA AMP for faster GPU training.
    - Supports CUDA throughput options: TF32, channels-last tensors, and loader prefetching.
    - Supports overfit-one-batch debugging.
"""

import os
import csv
import sys
import math
import time
import random
import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as io
import torchvision.transforms.functional as TF
from torchvision.io import ImageReadMode
from torch.utils.data import Dataset, DataLoader

class NullSummaryWriter:
    def __init__(self, *args, **kwargs):
        print("TensorBoard is not installed; continuing without event logs.")

    def add_scalar(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def make_summary_writer(logs_dir: Path):
    # Import lazily so Windows DataLoader worker processes do not import the
    # TensorBoard stack just to read image patches.
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(str(logs_dir))
    except ModuleNotFoundError:
        return NullSummaryWriter()

from Network.PhamHungSon_15_CNN_Network import (
    HomographyModel,
    train_loss_fn,
    rmse_loss_px,
    denormalize_prediction,
    DEFAULT_LABEL_SCALE,
)


# Do not generate __pycache__ files
sys.dont_write_bytecode = True


# =====================================================
# Reproducibility
# =====================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Good reproducibility defaults.
    # Turn benchmark off if you need exactly reproducible timing/results.
    torch.backends.cudnn.benchmark = True


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda")
        return torch.cuda.amp.autocast()
    return nullcontext()


def make_grad_scaler(device: torch.device, enabled: bool):
    scaler_enabled = bool(enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=scaler_enabled)
    return torch.cuda.amp.GradScaler(enabled=scaler_enabled)


def configure_cuda_performance(device: torch.device, allow_tf32: bool) -> None:
    if device.type != "cuda":
        return

    torch.backends.cudnn.benchmark = True

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)

    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def move_image_batch(x: torch.Tensor, device: torch.device, channels_last: bool) -> torch.Tensor:
    if channels_last and device.type == "cuda":
        return x.to(device, non_blocking=True, memory_format=torch.channels_last)
    return x.to(device, non_blocking=True)


def make_data_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))

    return DataLoader(**loader_kwargs)


def print_cuda_memory(device: torch.device, label: str) -> None:
    if device.type != "cuda":
        return

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    max_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    print(
        f"{label} CUDA memory | "
        f"allocated: {allocated:.2f} GiB | reserved: {reserved:.2f} GiB | "
        f"max allocated: {max_allocated:.2f} GiB | max reserved: {max_reserved:.2f} GiB"
    )


# =====================================================
# Path utilities
# =====================================================

def find_data_root(start_dir: Path) -> Path:
    """
    Searches upward for a project folder containing data/cnn.
    """
    start_dir = start_dir.resolve()

    for candidate in [start_dir, *start_dir.parents]:
        if (candidate / "data" / "cnn").exists():
            return candidate

    # Fallback: current script folder.
    return start_dir


def resolve_base_path(user_base_path: Optional[str]) -> Path:
    """
    Returns the folder containing Train_synthetic and Val_synthetic.
    """
    if user_base_path is not None:
        base_path = Path(user_base_path).expanduser().resolve()
    else:
        root_dir = Path(__file__).resolve().parent
        project_root = find_data_root(root_dir)
        base_path = project_root / "data" / "cnn" / "split"

    if not base_path.exists():
        raise FileNotFoundError(
            f"BasePath does not exist: {base_path}\n"
            "Pass it manually, for example:\n"
            "python PhamHungSon_15_CNN_Train_Sup.py --BasePath F:/YourProject/data/cnn/split"
        )

    return base_path


# =====================================================
# Dataset
# =====================================================

def read_h4_csv(csv_path: Path) -> Dict[str, List[float]]:
    """
    Reads H4.csv into:
        {filename: [dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4]}
    """
    labels: Dict[str, List[float]] = {}

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)

        for row_idx, row in enumerate(reader):
            if not row:
                continue

            filename = row[0].strip()

            # Skip header rows gracefully.
            try:
                values = [float(v) for v in row[1:9]]
            except ValueError:
                if row_idx == 0:
                    continue
                raise ValueError(f"Invalid numeric labels at row {row_idx + 1}: {row}")

            if len(values) != 8:
                raise ValueError(
                    f"Expected 8 label values at row {row_idx + 1}, got {len(values)}: {row}"
                )

            labels[filename] = values

    return labels


class HomographyPairDataset(Dataset):
    """
    Dataset for paired patch homography regression.

    Returns:
        x: Tensor [2, image_size, image_size], normalized to [-1, 1]
        y: Tensor [8], raw pixel offsets
        filename: str
    """

    def __init__(
        self,
        base_path: Path,
        split: str,
        image_size: int = 128,
        augment: bool = False,
        augment_strength: float = 1.0,
    ):
        super().__init__()

        self.base_path = Path(base_path)
        self.split = split
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.augment_strength = max(0.0, float(augment_strength))

        self.split_path = self.base_path / split
        self.pa_dir = self.split_path / "PA"
        self.pb_dir = self.split_path / "PB"
        self.label_path = self.split_path / "H4.csv"

        if not self.pa_dir.exists():
            raise FileNotFoundError(f"PA folder not found: {self.pa_dir}")
        if not self.pb_dir.exists():
            raise FileNotFoundError(f"PB folder not found: {self.pb_dir}")
        if not self.label_path.exists():
            raise FileNotFoundError(f"H4.csv not found: {self.label_path}")

        self.labels = read_h4_csv(self.label_path)

        pa_files = {p.name for p in self.pa_dir.iterdir() if p.is_file()}
        pb_files = {p.name for p in self.pb_dir.iterdir() if p.is_file()}
        label_files = set(self.labels.keys())

        self.files = sorted(pa_files & pb_files & label_files)

        self.pa_missing_labels = sorted(pa_files - label_files)
        self.pb_missing_labels = sorted(pb_files - label_files)
        self.labels_missing_pa = sorted(label_files - pa_files)
        self.labels_missing_pb = sorted(label_files - pb_files)

        if len(self.files) == 0:
            raise RuntimeError(
                f"No usable samples found in {self.split_path}. "
                "Check that PA, PB, and H4.csv use the same filenames."
            )

    def __len__(self) -> int:
        return len(self.files)

    def _read_grayscale(self, path: Path) -> torch.Tensor:
        img = io.read_image(str(path), mode=ImageReadMode.GRAY).float()

        if img.ndim != 3 or img.shape[0] != 1:
            raise ValueError(f"Expected grayscale image [1,H,W], got {tuple(img.shape)} from {path}")

        if img.shape[-2:] != (self.image_size, self.image_size):
            img = TF.resize(
                img,
                [self.image_size, self.image_size],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )

        return img

    @staticmethod
    def _flip_h4_horizontal(y: torch.Tensor) -> torch.Tensor:
        # Corner order is TL, BL, TR, BR. Mirroring swaps left/right corners
        # and reverses only the x displacement.
        corners = y.view(4, 2)
        flipped = corners[[2, 3, 0, 1]].clone()
        flipped[:, 0].mul_(-1.0)
        return flipped.reshape(-1)

    @staticmethod
    def _flip_h4_vertical(y: torch.Tensor) -> torch.Tensor:
        # Corner order is TL, BL, TR, BR. Mirroring swaps top/bottom corners
        # and reverses only the y displacement.
        corners = y.view(4, 2)
        flipped = corners[[1, 0, 3, 2]].clone()
        flipped[:, 1].mul_(-1.0)
        return flipped.reshape(-1)

    def _augment_pair(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        strength = self.augment_strength

        if random.random() < 0.50:
            x = torch.flip(x, dims=[2])
            y = self._flip_h4_horizontal(y)

        if random.random() < 0.50:
            x = torch.flip(x, dims=[1])
            y = self._flip_h4_vertical(y)

        shared_contrast = random.uniform(1.0 - 0.18 * strength, 1.0 + 0.18 * strength)
        shared_brightness = random.uniform(-0.12 * strength, 0.12 * strength)
        x = torch.clamp(x * shared_contrast + shared_brightness, -1.0, 1.0)

        # Let each patch have slightly different exposure/noise. This is common
        # in real stitching sequences and discourages memorizing textures.
        contrast = torch.tensor(
            [random.uniform(1.0 - 0.10 * strength, 1.0 + 0.10 * strength) for _ in range(2)],
            dtype=x.dtype,
        ).view(2, 1, 1)
        brightness = torch.tensor(
            [random.uniform(-0.08 * strength, 0.08 * strength) for _ in range(2)],
            dtype=x.dtype,
        ).view(2, 1, 1)
        x = torch.clamp(x * contrast + brightness, -1.0, 1.0)

        noise_std = random.uniform(0.0, 0.035 * strength)
        if noise_std > 0:
            x = torch.clamp(x + torch.randn_like(x) * noise_std, -1.0, 1.0)

        if random.random() < 0.20 * min(1.5, strength):
            kernel_size = random.choice([3, 5])
            sigma = random.uniform(0.2, 0.9 * strength)
            x = TF.gaussian_blur(x, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

        if random.random() < 0.25 * min(1.5, strength):
            erase_h = random.randint(max(4, self.image_size // 14), max(8, self.image_size // 5))
            erase_w = random.randint(max(4, self.image_size // 14), max(8, self.image_size // 5))
            top = random.randint(0, max(0, self.image_size - erase_h))
            left = random.randint(0, max(0, self.image_size - erase_w))
            fill = random.uniform(-0.15, 0.15)
            x[:, top:top + erase_h, left:left + erase_w] = fill

        return torch.clamp(x, -1.0, 1.0), y

    def __getitem__(self, index: int):
        filename = self.files[index]

        pa = self._read_grayscale(self.pa_dir / filename)
        pb = self._read_grayscale(self.pb_dir / filename)

        x = torch.cat([pa, pb], dim=0)

        # Normalize image input from [0,255] to [-1,1].
        x = (x / 255.0 - 0.5) / 0.5
        y = torch.tensor(self.labels[filename], dtype=torch.float32)

        if self.augment and self.augment_strength > 0:
            x, y = self._augment_pair(x, y)

        return x, y, filename


# =====================================================
# Debug utilities
# =====================================================

def print_dataset_debug(dataset: HomographyPairDataset) -> None:
    labels_np = np.array([dataset.labels[f] for f in dataset.files], dtype=np.float32)

    print(f"\n================ {dataset.split} DEBUG ================")
    print("Usable samples:", len(dataset))
    print("PA folder:", dataset.pa_dir)
    print("PB folder:", dataset.pb_dir)
    print("H4 labels:", dataset.label_path)
    print("PA files missing labels:", len(dataset.pa_missing_labels))
    print("PB files missing labels:", len(dataset.pb_missing_labels))
    print("Label rows missing PA:", len(dataset.labels_missing_pa))
    print("Label rows missing PB:", len(dataset.labels_missing_pb))
    print("Label min:", float(labels_np.min()))
    print("Label max:", float(labels_np.max()))
    print("Label mean:", float(labels_np.mean()))
    print("Label std:", float(labels_np.std()))
    print("First usable sample:", dataset.files[0])
    print("First label:", dataset.labels[dataset.files[0]])

    if len(dataset) < 50:
        print("WARNING: Very small dataset. Training will likely overfit.")

    print("====================================================\n")


# =====================================================
# Metrics
# =====================================================

class MetricAccumulator:
    def __init__(self):
        self.num_values = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0

        self.num_samples = 0
        self.sum_sample_mce = 0.0
        self.max_corner_error = 0.0

        self.acc_1 = 0
        self.acc_3 = 0
        self.acc_5 = 0
        self.outlier_10 = 0

    def update(self, pred_px: torch.Tensor, target_px: torch.Tensor) -> None:
        """
        pred_px and target_px: [B, 8]
        """
        diff = pred_px.detach().float() - target_px.detach().float()
        bsz = diff.shape[0]

        self.num_values += diff.numel()
        self.sum_abs += diff.abs().sum().item()
        self.sum_sq += (diff ** 2).sum().item()

        corner_errors = torch.linalg.norm(diff.view(-1, 4, 2), dim=2)  # [B,4]
        sample_mce = corner_errors.mean(dim=1)  # [B]

        self.num_samples += bsz
        self.sum_sample_mce += sample_mce.sum().item()
        self.max_corner_error = max(self.max_corner_error, corner_errors.max().item())

        self.acc_1 += (sample_mce < 1.0).sum().item()
        self.acc_3 += (sample_mce < 3.0).sum().item()
        self.acc_5 += (sample_mce < 5.0).sum().item()
        self.outlier_10 += (sample_mce > 10.0).sum().item()

    def compute(self) -> Dict[str, float]:
        if self.num_samples == 0 or self.num_values == 0:
            return {}

        rmse = math.sqrt(self.sum_sq / self.num_values)
        mae = self.sum_abs / self.num_values
        mce = self.sum_sample_mce / self.num_samples

        return {
            "rmse_px": rmse,
            "mae_px": mae,
            "mce_px": mce,
            "max_corner_error_px": self.max_corner_error,
            "acc_1px": self.acc_1 / self.num_samples * 100.0,
            "acc_3px": self.acc_3 / self.num_samples * 100.0,
            "acc_5px": self.acc_5 / self.num_samples * 100.0,
            "outlier_10px": self.outlier_10 / self.num_samples * 100.0,
        }


def print_metrics(title: str, metrics: Dict[str, float]) -> None:
    print(f"\n---------------- {title} METRICS ----------------")
    print(f"RMSE per coordinate: {metrics['rmse_px']:.4f} px")
    print(f"Mean Corner Error (MCE): {metrics['mce_px']:.4f} px")
    print(f"Mean Absolute Error (MAE): {metrics['mae_px']:.4f} px")
    print(f"Max Corner Error: {metrics['max_corner_error_px']:.4f} px")
    print(f"Accuracy @ 1.0px: {metrics['acc_1px']:.2f}%")
    print(f"Accuracy @ 3.0px: {metrics['acc_3px']:.2f}%")
    print(f"Accuracy @ 5.0px: {metrics['acc_5px']:.2f}%")
    print(f"Outlier Rate (>10px): {metrics['outlier_10px']:.2f}%")
    print("------------------------------------------------\n")


# =====================================================
# Train/eval operations
# =====================================================

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    label_scale: float,
    corner_loss_weight: float,
    scaler,
    amp_enabled: bool,
    channels_last: bool,
    grad_accum_steps: int,
    grad_clip: float = 5.0,
    print_every: int = 25,
    writer: Optional[object] = None,
) -> float:
    model.train()

    total_loss = 0.0
    num_batches = 0
    grad_accum_steps = max(1, int(grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (x, y, _) in enumerate(loader):
        x = move_image_batch(x, device, channels_last)
        y = y.to(device, non_blocking=True)

        with autocast_context(device, amp_enabled):
            pred_norm = model(x)
            raw_loss = train_loss_fn(
                pred_norm,
                y,
                label_scale=label_scale,
                corner_weight=corner_loss_weight,
            )
            loss = raw_loss / grad_accum_steps

        should_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(loader))

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if should_step:
                if grad_clip is not None and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if should_step:
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if scheduler is not None and should_step:
            scheduler.step()

        total_loss += raw_loss.item()
        num_batches += 1

        global_step = epoch * len(loader) + batch_idx

        if writer is not None:
            writer.add_scalar("train/loss_smoothl1_norm", raw_loss.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        if batch_idx % print_every == 0:
            with torch.no_grad():
                rmse_px = rmse_loss_px(pred_norm, y, label_scale=label_scale).item()

            print(
                f"Epoch {epoch + 1:03d} | "
                f"Batch {batch_idx + 1:04d}/{len(loader):04d} | "
                f"SmoothL1(norm): {raw_loss.item():.6f} | "
                f"RMSE(px): {rmse_px:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6e} | "
                f"Accum: {grad_accum_steps}"
            )

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_scale: float,
    title: str,
    amp_enabled: bool = False,
    channels_last: bool = False,
) -> Dict[str, float]:
    model.eval()

    metric_acc = MetricAccumulator()

    total_loss = 0.0
    num_batches = 0

    for x, y, _ in loader:
        x = move_image_batch(x, device, channels_last)
        y = y.to(device, non_blocking=True)

        with autocast_context(device, amp_enabled):
            pred_norm = model(x)
            loss = train_loss_fn(pred_norm, y, label_scale=label_scale)
        pred_px = denormalize_prediction(pred_norm, label_scale=label_scale)

        total_loss += loss.item()
        num_batches += 1

        metric_acc.update(pred_px, y)

    metrics = metric_acc.compute()
    metrics["smoothl1_norm"] = total_loss / max(1, num_batches)

    print_metrics(title, metrics)
    print(f"{title} SmoothL1 normalized loss: {metrics['smoothl1_norm']:.6f}")

    return metrics


def overfit_one_batch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_scale: float,
    steps: int = 1000,
    lr: float = 1e-3,
    channels_last: bool = False,
) -> None:
    """
    Debug test:
    The model should be able to overfit one small batch.
    If this cannot reach very low pixel RMSE, check labels/pairs/input format.

    IMPORTANT: We disable Dropout (model.eval()) so the network is deterministic
    and can memorize the batch. Dropout would prevent convergence on 8 samples.
    """
    # Switch to eval to disable Dropout/BatchNorm stochasticity.
    # This is intentional: we want the network to prove it CAN memorize
    # a small batch. If it can't even do this, there is a data bug.
    model.eval()

    x, y, filenames = next(iter(loader))
    x = move_image_batch(x, device, channels_last)
    y = y.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    print("\n================ OVERFIT ONE BATCH DEBUG ================")
    print("Batch size:", x.shape[0])
    print("First file:", filenames[0])
    print("Input shape:", tuple(x.shape))
    print("Target shape:", tuple(y.shape))
    print("Label scale:", label_scale)
    print("NOTE: Dropout is DISABLED for this test (model.eval()).")
    print("      If MCE doesn't go below 1px, there is a data pipeline bug.")

    for step in range(steps + 1):
        # Keep eval mode so gradients flow but Dropout is off.
        pred_norm = model(x)
        loss = train_loss_fn(pred_norm, y, label_scale=label_scale)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Mild clip to prevent any rare NaN explosion.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        if step % 100 == 0 or step == steps:
            with torch.no_grad():
                pred_px = denormalize_prediction(pred_norm, label_scale=label_scale)
                rmse = torch.sqrt(F.mse_loss(pred_px, y.float())).item()
                diff = pred_px - y
                mce = torch.linalg.norm(diff.view(-1, 4, 2), dim=2).mean().item()

            print(
                f"Step {step:04d} | "
                f"SmoothL1(norm): {loss.item():.6f} | "
                f"RMSE(px): {rmse:.4f} | "
                f"MCE(px): {mce:.4f}"
            )

    # Restore train mode for any subsequent code.
    model.train()

    with torch.no_grad():
        model.eval()
        pred_px = denormalize_prediction(model(x), label_scale=label_scale)
        model.train()

    print("\nFirst sample GT:")
    print(y[0].detach().cpu().numpy())
    print("First sample prediction:")
    print(pred_px[0].detach().cpu().numpy())
    print("=========================================================\n")


# =====================================================
# Checkpointing
# =====================================================

def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_mce: float,
    args,
    is_best: bool = False,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_mce": best_mce,
        "args": vars(args),
    }

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    temp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")

    try:
        torch.save(state, temp_path)
        os.replace(temp_path, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        if is_best:
            best_path = checkpoint_path.parent / "PhamHungSon_15_CNN_best_model.ckpt"
            torch.save(state, best_path)
            print(f"Saved best checkpoint: {best_path}")

    except Exception as exc:
        print(f"[WARNING] Failed to save checkpoint: {exc}")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def prune_old_checkpoints(checkpoint_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return

    epoch_checkpoints = sorted(
        checkpoint_dir.glob("PhamHungSon_15_CNN_epoch_*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for old_checkpoint in epoch_checkpoints[keep_last:]:
        try:
            old_checkpoint.unlink()
            print(f"Pruned old checkpoint: {old_checkpoint}")
        except Exception as exc:
            print(f"[WARNING] Could not prune checkpoint {old_checkpoint}: {exc}")


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: Optional[torch.device] = None,
) -> Tuple[int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_mce = float(checkpoint.get("best_mce", float("inf")))

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Resuming from epoch: {start_epoch + 1}")
    print(f"Best MCE so far: {best_mce:.4f}")

    return start_epoch, best_mce


# =====================================================
# Main
# =====================================================

def build_optimizer(model: torch.nn.Module, name: str, lr: float, weight_decay: float):
    name = name.lower()

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=weight_decay,
        )

    raise ValueError("Optimizer must be Adam, AdamW, or SGD.")


def read_checkpoint_args(checkpoint_path: Optional[str], device: torch.device) -> Dict[str, object]:
    if checkpoint_path is None:
        return {}

    path = Path(checkpoint_path)
    if not path.exists():
        return {}

    try:
        checkpoint = torch.load(path, map_location=device)
    except Exception as exc:
        print(f"[WARNING] Could not inspect checkpoint args: {exc}")
        return {}

    return dict(checkpoint.get("args", {}) or {})


def resolve_auto_setting(
    requested: str,
    checkpoint_args: Dict[str, object],
    key: str,
    scratch_default: str,
    legacy_checkpoint_default: str,
) -> str:
    if requested != "auto":
        return requested

    if checkpoint_args:
        return str(checkpoint_args.get(key, legacy_checkpoint_default))

    return scratch_default


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--BasePath", type=str, default=None, help="Path to data/cnn/split")
    parser.add_argument("--NumEpochs", type=int, default=100)
    parser.add_argument("--MiniBatchSize", type=int, default=16)
    parser.add_argument("--GradAccumSteps", type=int, default=1, help="Accumulate gradients across N mini-batches. Effective batch = MiniBatchSize * GradAccumSteps.")
    parser.add_argument("--ImageSize", type=int, default=128)

    parser.add_argument("--Backbone", type=str, default="ConvNet", choices=["ConvNet", "SuperNet", "ResNet18"])
    parser.add_argument(
        "--InputMode",
        type=str,
        default="auto",
        choices=["auto", "basic", "coord", "pairdiff", "pairdiff_coord"],
        help="auto uses pairdiff_coord for new runs and legacy basic when resuming old checkpoints.",
    )
    parser.add_argument(
        "--OutputActivation",
        type=str,
        default="auto",
        choices=["auto", "linear", "tanh"],
        help="auto uses linear for new runs and legacy tanh when resuming old checkpoints.",
    )
    parser.add_argument("--Dropout", type=float, default=0.35)

    parser.add_argument("--Optimizer", type=str, default="AdamW", choices=["Adam", "AdamW", "SGD"])
    parser.add_argument("--LR", type=float, default=3e-4)
    parser.add_argument("--WeightDecay", type=float, default=5e-4)
    parser.add_argument("--LabelScale", type=float, default=42.0)
    parser.add_argument(
        "--CornerLossWeight",
        type=float,
        default=0.15,
        help="Adds normalized corner-distance loss to SmoothL1. Use 0 to disable.",
    )
    parser.add_argument("--Amp", action=argparse.BooleanOptionalAction, default=True, help="Use CUDA AMP when available.")
    parser.add_argument("--ChannelsLast", action=argparse.BooleanOptionalAction, default=True, help="Use channels-last tensors on CUDA for faster convolution throughput.")
    parser.add_argument("--AllowTF32", action=argparse.BooleanOptionalAction, default=True, help="Allow TF32 matmul/convolution on RTX GPUs for faster training.")

    parser.add_argument("--NumWorkers", type=int, default=0, help="Use 0 on Windows if multiprocessing causes issues.")
    parser.add_argument("--PrefetchFactor", type=int, default=2, help="Batches prefetched per DataLoader worker when NumWorkers > 0.")
    parser.add_argument("--PersistentWorkers", action=argparse.BooleanOptionalAction, default=False, help="Keep training DataLoader workers alive between epochs when NumWorkers > 0. Faster, but uses more RAM.")
    parser.add_argument("--Seed", type=int, default=42)

    parser.add_argument("--CheckPointPath", type=str, default="PhamHungSon_15_CNN_CheckpointsSupRegularized")
    parser.add_argument("--LogsPath", type=str, default="PhamHungSon_15_CNN_LogsSupRegularized")
    parser.add_argument("--Resume", type=str, default=None, help="Path to checkpoint to resume from.")

    parser.add_argument("--SaveEvery", type=int, default=10)
    parser.add_argument("--KeepLastCheckpoints", type=int, default=5, help="Keep only the latest N epoch checkpoints. The best checkpoint is always kept. <=0 disables pruning.")
    parser.add_argument("--PrintEvery", type=int, default=25)
    parser.add_argument("--GradClip", type=float, default=5.0)

    parser.add_argument("--NoAugment", action="store_true", help="Disable brightness/contrast augmentation.")
    parser.add_argument("--AugmentStrength", type=float, default=1.0, help="Scale online augmentation intensity. Try 0.5-1.5.")
    parser.add_argument("--EvalTrainEvery", type=int, default=5, help="Evaluate train set every N epochs. Use 0 to skip full train-set evaluation.")
    parser.add_argument("--EarlyStopPatience", type=int, default=20, help="Stop if validation MCE does not improve for N epochs. <=0 disables.")
    parser.add_argument("--MinDelta", type=float, default=1e-3, help="Minimum validation MCE improvement in pixels.")
    parser.add_argument("--MemoryReportEvery", type=int, default=1, help="Print CUDA memory statistics every N epochs. <=0 disables.")
    parser.add_argument("--OverfitOneBatch", action="store_true", help="Run overfit-one-batch debug test and exit.")
    parser.add_argument("--OverfitSteps", type=int, default=1000)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.Seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_cuda_performance(device, args.AllowTF32)

    base_path = resolve_base_path(args.BasePath)
    checkpoint_args = read_checkpoint_args(args.Resume, device)
    args.InputMode = resolve_auto_setting(
        requested=args.InputMode,
        checkpoint_args=checkpoint_args,
        key="InputMode",
        scratch_default="pairdiff_coord",
        legacy_checkpoint_default="basic",
    )
    args.OutputActivation = resolve_auto_setting(
        requested=args.OutputActivation,
        checkpoint_args=checkpoint_args,
        key="OutputActivation",
        scratch_default="linear",
        legacy_checkpoint_default="tanh",
    )

    script_dir = Path(__file__).resolve().parent
    checkpoint_dir = Path(args.CheckPointPath)
    logs_dir = Path(args.LogsPath)

    if not checkpoint_dir.is_absolute():
        checkpoint_dir = script_dir / checkpoint_dir

    if not logs_dir.is_absolute():
        logs_dir = script_dir / logs_dir

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("\n================ TRAIN CONFIG ================")
    print("Device:", device)
    print("BasePath:", base_path)
    print("Backbone:", args.Backbone)
    print("InputMode:", args.InputMode)
    print("OutputActivation:", args.OutputActivation)
    print("Epochs:", args.NumEpochs)
    print("MiniBatchSize:", args.MiniBatchSize)
    print("GradAccumSteps:", max(1, args.GradAccumSteps))
    print("EffectiveBatchSize:", args.MiniBatchSize * max(1, args.GradAccumSteps))
    print("ImageSize:", args.ImageSize)
    print("Optimizer:", args.Optimizer)
    print("LR:", args.LR)
    print("WeightDecay:", args.WeightDecay)
    print("LabelScale:", args.LabelScale)
    print("CornerLossWeight:", args.CornerLossWeight)
    print("AMP:", bool(args.Amp and device.type == "cuda"))
    print("ChannelsLast:", bool(args.ChannelsLast and device.type == "cuda"))
    print("AllowTF32:", bool(args.AllowTF32 and device.type == "cuda"))
    print("NumWorkers:", args.NumWorkers)
    print("PrefetchFactor:", args.PrefetchFactor if args.NumWorkers > 0 else "disabled")
    print("PersistentWorkers:", bool(args.PersistentWorkers and args.NumWorkers > 0))
    print("Augment:", not args.NoAugment)
    print("AugmentStrength:", 0.0 if args.NoAugment else args.AugmentStrength)
    print("SaveEvery:", args.SaveEvery)
    print("KeepLastCheckpoints:", args.KeepLastCheckpoints)
    print("EvalTrainEvery:", args.EvalTrainEvery)
    print("EarlyStopPatience:", args.EarlyStopPatience)
    print("Checkpoint dir:", checkpoint_dir)
    print("Logs dir:", logs_dir)
    print("================================================\n")

    train_dataset = HomographyPairDataset(
        base_path=base_path,
        split="Train_synthetic",
        image_size=args.ImageSize,
        augment=not args.NoAugment,
        augment_strength=args.AugmentStrength,
    )

    val_dataset = HomographyPairDataset(
        base_path=base_path,
        split="Val_synthetic",
        image_size=args.ImageSize,
        augment=False,
        augment_strength=0.0,
    )

    print_dataset_debug(train_dataset)
    print_dataset_debug(val_dataset)

    pin_memory = device.type == "cuda"
    train_loader = make_data_loader(
        dataset=train_dataset,
        batch_size=args.MiniBatchSize,
        shuffle=True,
        num_workers=args.NumWorkers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=args.PersistentWorkers,
        prefetch_factor=args.PrefetchFactor,
    )

    val_loader = make_data_loader(
        dataset=val_dataset,
        batch_size=args.MiniBatchSize,
        shuffle=False,
        num_workers=args.NumWorkers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=False,
        prefetch_factor=args.PrefetchFactor,
    )

    # Separate non-augmented train loader for honest training-set evaluation.
    train_eval_dataset = HomographyPairDataset(
        base_path=base_path,
        split="Train_synthetic",
        image_size=args.ImageSize,
        augment=False,
        augment_strength=0.0,
    )

    train_eval_loader = make_data_loader(
        dataset=train_eval_dataset,
        batch_size=args.MiniBatchSize,
        shuffle=False,
        num_workers=args.NumWorkers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=False,
        prefetch_factor=args.PrefetchFactor,
    )

    model = HomographyModel(
        backbone=args.Backbone,
        dropout=args.Dropout,
        input_mode=args.InputMode,
        output_activation=args.OutputActivation,
    ).to(device)
    if args.ChannelsLast and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    print_cuda_memory(device, "After model setup")

    optimizer = build_optimizer(
        model=model,
        name=args.Optimizer,
        lr=args.LR,
        weight_decay=args.WeightDecay,
    )

    # ReduceLROnPlateau: halves LR when val MCE stops improving.
    # This is much better than CosineAnnealing which decays to near-zero
    # after only a few epochs and prevents the model from escaping plateaus.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=7,
        min_lr=1e-6,
    )

    start_epoch = 0
    best_mce = float("inf")

    if args.Resume is not None:
        start_epoch, best_mce = load_checkpoint(
            checkpoint_path=Path(args.Resume),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    writer = make_summary_writer(logs_dir)
    scaler = make_grad_scaler(device, args.Amp)
    amp_enabled = bool(args.Amp and device.type == "cuda")

    if args.OverfitOneBatch:
        overfit_loader = make_data_loader(
            dataset=train_eval_dataset,
            batch_size=min(args.MiniBatchSize, 8),
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
            drop_last=False,
        )
        overfit_one_batch(
            model=model,
            loader=overfit_loader,
            device=device,
            label_scale=args.LabelScale,
            steps=args.OverfitSteps,
            lr=max(args.LR, 1e-3),
            channels_last=args.ChannelsLast,
        )
        return

    epochs_without_improvement = 0

    for epoch in range(start_epoch, args.NumEpochs):
        print(f"\n================ EPOCH {epoch + 1}/{args.NumEpochs} ================")

        start_time = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=None,   # ReduceLROnPlateau is stepped on val MCE below
            device=device,
            epoch=epoch,
            label_scale=args.LabelScale,
            corner_loss_weight=args.CornerLossWeight,
            scaler=scaler,
            amp_enabled=amp_enabled,
            channels_last=args.ChannelsLast,
            grad_accum_steps=args.GradAccumSteps,
            grad_clip=args.GradClip,
            print_every=args.PrintEvery,
            writer=writer,
        )

        print(f"Epoch {epoch + 1} average train SmoothL1(norm): {train_loss:.6f}")

        if args.EvalTrainEvery > 0 and (epoch + 1) % args.EvalTrainEvery == 0:
            train_metrics = evaluate(
                model=model,
                loader=train_eval_loader,
                device=device,
                label_scale=args.LabelScale,
                title="TRAIN",
                amp_enabled=amp_enabled,
                channels_last=args.ChannelsLast,
            )

            writer.add_scalar("train_eval/rmse_px", train_metrics["rmse_px"], epoch)
            writer.add_scalar("train_eval/mce_px", train_metrics["mce_px"], epoch)

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            label_scale=args.LabelScale,
            title="VALIDATION",
            amp_enabled=amp_enabled,
            channels_last=args.ChannelsLast,
        )

        writer.add_scalar("validation/smoothl1_norm", val_metrics["smoothl1_norm"], epoch)
        writer.add_scalar("validation/rmse_px", val_metrics["rmse_px"], epoch)
        writer.add_scalar("validation/mce_px", val_metrics["mce_px"], epoch)
        writer.add_scalar("validation/mae_px", val_metrics["mae_px"], epoch)
        writer.add_scalar("validation/outlier_10px", val_metrics["outlier_10px"], epoch)

        current_mce = val_metrics["mce_px"]
        is_best = current_mce < (best_mce - args.MinDelta)

        # Step ReduceLROnPlateau on validation MCE.
        scheduler.step(current_mce)
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")

        if is_best:
            best_mce = current_mce
            epochs_without_improvement = 0
            print(f"New best validation MCE: {best_mce:.4f} px")
        else:
            epochs_without_improvement += 1
            print(
                f"No validation improvement for {epochs_without_improvement} epoch(s). "
                f"Best MCE remains {best_mce:.4f} px"
            )

        if (epoch + 1) % args.SaveEvery == 0 or is_best:
            ckpt_path = checkpoint_dir / f"PhamHungSon_15_CNN_epoch_{epoch + 1:03d}.ckpt"
            save_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_mce=best_mce,
                args=args,
                is_best=is_best,
            )
            prune_old_checkpoints(checkpoint_dir, args.KeepLastCheckpoints)

        elapsed = time.time() - start_time
        print(f"Epoch time: {elapsed:.2f} seconds")
        if args.MemoryReportEvery > 0 and (epoch + 1) % args.MemoryReportEvery == 0:
            print_cuda_memory(device, f"Epoch {epoch + 1}")

        writer.flush()

        if args.EarlyStopPatience > 0 and epochs_without_improvement >= args.EarlyStopPatience:
            print(
                f"Early stopping triggered after {epochs_without_improvement} epochs "
                "without validation MCE improvement."
            )
            break

    writer.close()
    print("\nTraining finished.")
    print(f"Best validation MCE: {best_mce:.4f} px")
    print(f"Best checkpoint: {checkpoint_dir / 'PhamHungSon_15_CNN_best_model.ckpt'}")


if __name__ == "__main__":
    main()
