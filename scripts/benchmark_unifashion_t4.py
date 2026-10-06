"""Short real-data throughput/stability check; never writes a training checkpoint."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader
from scripts.train_unifashion_sanity import build_model, file_hash, train_batch
from src.unifashion_sanity_data import UniFashionSanityDataset, collate
from src.unifashion_training import make_batches


def measure_updates(model, loader, optimizer, scaler, config, device, warmup, steps):
    """Include image loading and retry time, exclude warm-up and model loading."""
    if warmup < 0 or steps < 1:
        raise ValueError("warmup >= 0 and steps >= 1 required")
    iterator = iter(loader)
    rows = []
    for index in range(warmup + steps):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        batch = next(iterator)
        metrics = train_batch(model, batch, optimizer, scaler, config, device,
                              lambda event: print(json.dumps({"overflow": event}), flush=True))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row = {"update": index + 1, "warmup": index < warmup,
               "samples": len(batch["text_input"]),
               "seconds_including_data": time.perf_counter() - started, **metrics}
        rows.append(row)
        print(json.dumps(row), flush=True)
    measured = rows[warmup:]
    seconds = sum(r["seconds_including_data"] for r in measured)
    samples = sum(r["samples"] for r in measured)
    return {"measured_updates": steps, "samples_per_second": samples / seconds,
            "seconds_per_update": seconds / steps,
            "amp_retries_including_warmup": sum(r["amp_retries"] for r in rows), "updates": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()
    if args.output.exists():
        p.error("Choose a new output file to preserve previous measurements")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"status": "failed", "precision": args.precision, "batch_size": args.batch_size,
              "purpose": "Short throughput/stability check, not full training or retrieval evaluation",
              "checkpoint_written": False}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Run this benchmark on your Colab GPU, after stopping the training process")
        config = json.loads((ROOT / "configs/unifashion_finetune_colab.json").read_text())
        config.update(precision=args.precision, batch_size=args.batch_size)
        audit = json.loads((args.data_dir / "data_audit.json").read_text())
        if file_hash(args.data_dir / "train.csv") != audit["files"]["train.csv"]:
            raise ValueError("Training CSV hash mismatch")
        torch.manual_seed(config["seed"])
        torch.set_num_threads(2)
        device = torch.device("cuda")
        dataset = UniFashionSanityDataset(args.data_dir / "train.csv", args.images_dir, config,
                                         require_unique_targets=False)
        batches, dropped = make_batches(dataset.data["target"].tolist(), args.batch_size, config["seed"], 0)
        if args.steps < 1 or args.warmup < 0 or args.steps + args.warmup > len(batches):
            raise ValueError("Invalid benchmark length")
        loader = DataLoader(dataset, batch_sampler=batches[:args.steps + args.warmup], collate_fn=collate,
                            generator=torch.Generator().manual_seed(config["seed"]), num_workers=0)
        print("Loading model for", args.precision, "batch", args.batch_size, flush=True)
        model, _ = build_model(ROOT / "third_party/UniFashion", ROOT / "checkpoints/unifashion", config, device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
            lr=config["learning_rate"], betas=tuple(config["betas"]), eps=config["eps"],
            weight_decay=config["weight_decay"], foreach=False)
        scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16",
                                     init_scale=config["grad_scaler_init_scale"])
        torch.cuda.reset_peak_memory_stats()
        result.update(measure_updates(model, loader, optimizer, scaler, config, device, args.warmup, args.steps))
        result.update(status="benchmark_passed", gpu=torch.cuda.get_device_name(),
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            planned_batches_per_epoch=len(batches), dropped_queries=len(dropped),
            estimated_train_hours_per_epoch=len(batches)*result["seconds_per_update"]/3600,
            estimate_excludes="Model loading, validation, checkpoint writes, and longer-run variability",
            benchmark_lr="Fixed configured max LR for a short stability probe; full training retains OneCycle",
            config=config)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("updates", "config")}, indent=2), flush=True)
    if result["status"] != "benchmark_passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
