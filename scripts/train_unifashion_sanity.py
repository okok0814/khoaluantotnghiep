"""One genuine UniFashion CIR fine-tuning epoch on a small FashionIQ subset.

Run --inspect-only on a laptop to audit domain checkpoint compatibility without
allocating EVA-G. Full forward/backward training requires a CUDA GPU.
"""
import argparse
from contextlib import nullcontext, redirect_stdout, redirect_stderr
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.unifashion_runtime import build_model
from src.unifashion_sanity_data import UniFashionSanityDataset, collate


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def device_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def check_losses(losses):
    if set(losses) != {"loss_itc", "loss_itm", "loss_ttc"}:
        raise RuntimeError(f"Unexpected official losses: {list(losses)}")
    if any(value.ndim != 0 or not torch.isfinite(value) for value in losses.values()):
        raise FloatingPointError(f"Non-finite or nonscalar losses: {losses}")
    return sum(losses.values())


def autocast_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device.type, dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[precision])


def train_batch(model, batch, optimizer, scaler, config, device, on_overflow=None):
    """Count a batch only after a finite update; retry AMP overflow on that batch.

    Restoring Torch RNG makes dropout identical on retry. GradScaler skips the
    invalid optimizer update, lowers its scale, and clears its unscale bookkeeping.
    Persistent bad gradients are an error, never a successful epoch.
    """
    model.train()
    model.visual_encoder.eval()
    trainable = [p for p in model.parameters() if p.requires_grad]
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    batch = device_batch(batch, device)
    max_retries = config.get("max_amp_retries", 10)
    for attempt in range(max_retries + 1):
        if attempt:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
        optimizer.zero_grad(set_to_none=True)
        scale_before = scaler.get_scale()
        with autocast_context(device, config["precision"]):
            losses = model(batch)
            loss = check_losses(losses)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        bad = [name for name, p in model.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad:
            event = {"attempt": attempt + 1, "loss_scale": scale_before,
                     "nonfinite_gradient_parameters": bad,
                     "losses": {key: value.item() for key, value in losses.items()},
                     "optimizer_update_applied": False}
            if scaler.is_enabled():
                # unscale_ recorded non-finite gradients, so step MUST skip.
                scaler.step(optimizer)
                scaler.update()
            event["next_loss_scale"] = scaler.get_scale()
            if on_overflow:
                on_overflow(event)
            optimizer.zero_grad(set_to_none=True)
            del losses, loss
            if not scaler.is_enabled() or attempt == max_retries:
                raise FloatingPointError(
                    f"Non-finite gradients persist after {attempt + 1} attempts; "
                    f"scale={scale_before}, parameters={bad[:8]}. "
                    "See amp_overflows.jsonl. Rerun with --precision fp32 to diagnose "
                    "without FP16; do not disable error_if_nonfinite.")
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, config["max_grad_norm"], error_if_nonfinite=True)
        if grad_norm.item() <= 0:
            raise RuntimeError("No gradient reached trainable model parameters")
        if any(p.grad is not None for p in model.visual_encoder.parameters()):
            raise RuntimeError("Frozen visual backbone unexpectedly received gradients")
        scaler.step(optimizer)
        scaler.update()
        return {**{key: value.item() for key, value in losses.items()},
                "loss": loss.item(), "grad_norm_before_clip": grad_norm.item(),
                "amp_retries": attempt, "loss_scale_before": scale_before,
                "loss_scale_after": scaler.get_scale()}


@torch.no_grad()
def validation_loss(model, loader, device, precision="fp16"):
    model.eval()
    totals, count = {}, 0
    for batch in loader:
        n = batch["image"].shape[0]
        with autocast_context(device, precision):
            losses = model(device_batch(batch, device))
            check_losses(losses)
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + value.item() * n
        count += n
    return {key: value / count for key, value in totals.items()}


def verify_subset(data_dir, images_dir):
    audit = json.loads((data_dir / "data_audit.json").read_text())
    for split in ("train", "val"):
        if file_hash(data_dir / f"{split}.csv") != audit["splits"][split]["subset_sha256"]:
            raise ValueError(f"{split}.csv changed after data audit")
    for image in audit["selected_images"].values():
        if file_hash(images_dir / image["file"]) != image["sha256"]:
            raise ValueError(f"Image changed after data audit: {image['file']}")


def execute(args, output):
    config = json.loads(args.config.read_text())
    if args.precision:
        config["precision"] = args.precision
    config.setdefault("grad_scaler_init_scale", 256.0)
    config.setdefault("max_amp_retries", 10)
    if config["epochs"] != 1 or config["batch_size"] < 2 or not config["freeze_vit"]:
        raise ValueError("Sanity config requires one epoch, batch_size >= 2, frozen vision")
    if config["precision"] not in ("fp16", "bf16", "fp32"):
        raise ValueError("precision must be fp16, bf16, or fp32")
    if not math.isfinite(config["grad_scaler_init_scale"]) or config["grad_scaler_init_scale"] <= 0:
        raise ValueError("grad_scaler_init_scale must be finite and positive")
    if not isinstance(config["max_amp_retries"], int) or config["max_amp_retries"] < 0:
        raise ValueError("max_amp_retries must be a nonnegative integer")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.inspect_only and device.type != "cuda":
        raise RuntimeError("Full UniFashion run requires CUDA. Use the Colab/Kaggle notebook; --inspect-only is CPU-safe.")
    if not args.inspect_only and config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; use fp16 or fp32")
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "transformers", "timm", "accelerate", "numpy", "pandas", "Pillow")}
    write_json(output / "environment.json", {"python": sys.version, "packages": versions,
               "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None})
    write_json(output / "config.json", config)
    write_json(output / "status.json", {"status": "running", "inspection_only": args.inspect_only})
    manifest = json.loads((args.assets / "download_manifest.json").read_text())
    if file_hash(args.assets / "pretrain/none_lora_0.pt") != "f272d3a3cc91ae809f844d3e258f6a872be405cb203a550c15c4510c7280b541":
        raise ValueError("Domain checkpoint SHA-256 mismatch")
    write_json(output / "download_manifest.json", manifest)
    print("Loading official UniFashion domain-pretraining weights", flush=True)
    model, report = build_model(args.upstream, args.assets, config, device, inspect_only=args.inspect_only)
    write_json(output / "checkpoint_load.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if args.inspect_only:
        result = {"status": "checkpoint_inspection_passed", "full_model_forward_tested": False,
                  "epoch_completed": False, "note": "EVA-G is meta-only; checkpoint tensors loaded for key/shape audit"}
        write_json(output / "status.json", result)
        print(json.dumps(result), flush=True)
        return
    verify_subset(args.data_dir, args.images_dir)
    train = UniFashionSanityDataset(args.data_dir / "train.csv", args.images_dir, config)
    val = UniFashionSanityDataset(args.data_dir / "val.csv", args.images_dir, config)
    train_ids = set(train.data["candidate"]) | set(train.data["target"])
    val_ids = set(val.data["candidate"]) | set(val.data["target"])
    if train_ids & val_ids:
        raise ValueError("Train/validation image leakage in sanity subsets")
    batch_size = config["batch_size"]
    if any(len(dataset) < batch_size or len(dataset) % batch_size for dataset in (train, val)):
        raise ValueError("Subset size must be divisible by batch_size; no examples are silently dropped")
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator,
                              num_workers=config["num_workers"], collate_fn=collate)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False,
                            num_workers=config["num_workers"], collate_fn=collate)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"], betas=tuple(config["betas"]),
                                  eps=config["eps"], weight_decay=config["weight_decay"], foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=config["precision"] == "fp16",
                                 init_scale=config["grad_scaler_init_scale"])
    initial_probe = model.query_tokens.detach().cpu().clone()
    before = validation_loss(model, val_loader, device, config["precision"])
    write_json(output / "validation_before.json", before)
    print("Validation smoke loss before:", before, flush=True)
    metrics, consumed = [], 0
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    for step, batch in enumerate(train_loader, 1):
        def log_overflow(event):
            event.update(epoch=1, step=step)
            with (output / "amp_overflows.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
            print(json.dumps({"amp_overflow_retry": event}), flush=True)

        step_metrics = train_batch(model, batch, optimizer, scaler, config, device, log_overflow)
        consumed += len(batch["text_input"])
        record = {"epoch": 1, "step": step, "samples_seen": consumed,
                  **step_metrics,
                  "lr": optimizer.param_groups[0]["lr"], "elapsed_s": time.monotonic() - start}
        metrics.append(record)
        with (output / "train_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record))
            writer.writeheader()
            writer.writerows(metrics)
        print(json.dumps(record), flush=True)
    if consumed != len(train) or torch.equal(initial_probe, model.query_tokens.detach().cpu()):
        raise RuntimeError("Incomplete epoch or pretrained query tokens did not update")
    after = validation_loss(model, val_loader, device, config["precision"])
    write_json(output / "validation_after.json", after)
    checkpoint_dir = ROOT / "checkpoints" / output.name
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = checkpoint_dir / "last.pt"
    # Frozen EVA weights are referenced by the manifest instead of duplicated.
    torch.save({"model": {key: value.detach().cpu() for key, value in model.state_dict().items()
                           if not key.startswith("visual_encoder.")},
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "epoch": 1, "global_step": len(metrics), "config": config,
                "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state(),
                "loader_rng_state": generator.get_state()}, checkpoint_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    # Perturb a trained parameter to make the reload check meaningful.
    with torch.no_grad():
        model.query_tokens.add_(0.1)
    incompatible = model.load_state_dict(saved["model"], strict=False)
    if incompatible.unexpected_keys or any(not key.startswith("visual_encoder.") for key in incompatible.missing_keys):
        raise RuntimeError(f"Reload mismatch: {incompatible}")
    optimizer.load_state_dict(saved["optimizer"])
    scaler.load_state_dict(saved["scaler"])
    reloaded = validation_loss(model, val_loader, device, config["precision"])
    if any(not np.isclose(after[key], reloaded[key], rtol=1e-4, atol=1e-5) for key in after):
        raise RuntimeError(f"Reload changed evaluation losses: {after} vs {reloaded}")
    result = {"status": "sanity_passed", "epoch_completed": True, "epochs": 1,
              "train_samples": consumed, "val_samples": len(val), "optimizer_steps": len(metrics),
              "amp_overflow_retries": sum(row["amp_retries"] for row in metrics),
              "precision": config["precision"],
              "query_tokens_changed": True, "reload_verified": True,
              "validation_before": before, "validation_after": after, "validation_reloaded": reloaded,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "checkpoint": str(checkpoint_path), "checkpoint_sha256": file_hash(checkpoint_path),
              "note": "Sanity validation loss only; not FashionIQ Recall@K, not paper reproduction"}
    write_json(output / "status.json", result)
    print(json.dumps(result, indent=2), flush=True)


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/unifashion_sanity.json")
    parser.add_argument("--upstream", type=Path, default=ROOT / "third_party/UniFashion")
    parser.add_argument("--assets", type=Path, default=ROOT / "checkpoints/unifashion")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "reports/week7_unifashion/data")
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"),
                        help="Override config; fp32 disables autocast and loss scaling")
    args = parser.parse_args()
    if not args.inspect_only and args.images_dir is None:
        parser.error("--images-dir is required for training")
    output = args.output or ROOT / "reports/week7_unifashion" / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "run.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                execute(args, output)
            except Exception as error:
                traceback.print_exc()
                write_json(output / "status.json", {"status": "failed", "error": str(error), "epoch_completed": False})
                raise


if __name__ == "__main__":
    main()
