"""Full FashionIQ training with the upstream optimizer/schedule and resumable runs."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import csv
import importlib.metadata
import json
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
from scripts.train_unifashion_sanity import (
    Tee, build_model, file_hash, train_batch, validation_loss, write_json,
)
from src.unifashion_sanity_data import UniFashionSanityDataset, collate
from src.unifashion_training import atomic_save, load_trainable_state, make_batches, model_state


def write_csv(path, rows):
    if not rows:
        path.unlink(missing_ok=True)
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_data(data_dir, images_dir):
    audit = json.loads((data_dir / "data_audit.json").read_text())
    for name, expected in audit["files"].items():
        if file_hash(data_dir / name) != expected:
            raise ValueError(f"Data changed since audit: {name}")
    manifest = json.loads((data_dir / "images_manifest.json").read_text())
    for i, record in enumerate(manifest.values(), 1):
        if file_hash(images_dir / record["file"]) != record["sha256"]:
            raise ValueError(f"Image changed since audit: {record['file']}")
        if i % 10000 == 0:
            print(f"Verified {i}/{len(manifest)} images", flush=True)
    return audit


def execute(args):
    out = args.output
    config = json.loads(args.config.read_text())
    if config["batch_size"] < 2 or not config["freeze_vit"] or config["num_workers"] != 0:
        raise ValueError("Require batch_size >= 2, frozen vision, workers=0 for deterministic resume")
    if config["scheduler"] != "one_cycle" or config["epochs"] <= config["warmup_epochs"]:
        raise ValueError("Require OneCycle schedule and epochs > warmup_epochs")
    if config["precision"] not in ("fp32", "fp16", "bf16"):
        raise ValueError("Unsupported precision")
    if not torch.cuda.is_available():
        raise RuntimeError("Use the GPU notebook; this full EVA-G model cannot train on this CPU laptop")
    device = torch.device("cuda")
    if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support BF16")
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= config["epochs"]:
        raise ValueError("stop-after-epoch must be within the configured schedule")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(2)
    audit = verify_data(args.data_dir, args.images_dir)
    train = UniFashionSanityDataset(args.data_dir / "train.csv", args.images_dir, config,
                                   require_unique_targets=False)
    monitor = UniFashionSanityDataset(args.data_dir / "monitor.csv", args.images_dir, config)
    if len(monitor) % config["batch_size"]:
        raise ValueError("Monitor size must be divisible by batch_size")
    targets = train.data["target"].tolist()
    plans = [make_batches(targets, config["batch_size"], config["seed"], e)
             for e in range(config["epochs"])]
    if not all(batches for batches, _ in plans):
        raise ValueError("No complete unique-target batches")
    total_steps = sum(len(batches) for batches, _ in plans)
    source_files = [Path(__file__), ROOT / "src/unifashion_training.py",
                    ROOT / "src/unifashion_runtime.py", ROOT / "src/unifashion_sanity_data.py",
                    ROOT / "src/fashioniq_unifashion_dataset.py", ROOT / "scripts/train_unifashion_sanity.py"]
    fingerprint = {"data_audit": file_hash(args.data_dir / "data_audit.json"),
                   "code": {p.name: file_hash(p) for p in source_files},
                   "assets_manifest": file_hash(args.assets / "download_manifest.json")}
    if file_hash(args.assets / "pretrain/none_lora_0.pt") != "f272d3a3cc91ae809f844d3e258f6a872be405cb203a550c15c4510c7280b541":
        raise ValueError("Domain checkpoint SHA-256 mismatch")
    saved = None
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True)
        if saved["config"] != config or saved["fingerprint"] != fingerprint:
            raise ValueError("Resume config/data/code/assets changed; use the original bundle/config")
    write_json(out / "config.json", config)
    write_json(out / "data_audit.json", audit)
    write_json(out / "batch_plan.json", [{"epoch": e + 1, "batches": len(b),
               "samples": len(b) * config["batch_size"], "dropped_row_indices": d}
              for e, (b, d) in enumerate(plans)])
    write_json(out / "environment.json", {"python": sys.version, "torch": str(torch.__version__),
               "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
               "packages": {name: importlib.metadata.version(name) for name in
                            ("torch", "torchvision", "transformers", "timm", "accelerate", "numpy", "pandas", "Pillow")},
               "fingerprint": fingerprint})
    print("Loading UniFashion + frozen EVA-G; training precision:", config["precision"], flush=True)
    model, report = build_model(args.upstream, args.assets, config, device)
    write_json(out / "checkpoint_load.json", report)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"], betas=tuple(config["betas"]), eps=config["eps"],
        weight_decay=config["weight_decay"], foreach=False)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["learning_rate"],
        total_steps=total_steps, pct_start=config["warmup_epochs"] / config["epochs"],
        div_factor=config["div_factor"], final_div_factor=config["final_div_factor"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["precision"] == "fp16",
                                 init_scale=config["grad_scaler_init_scale"])
    monitor_loader = DataLoader(monitor, batch_size=config["batch_size"], shuffle=False,
        collate_fn=collate, num_workers=0, generator=torch.Generator().manual_seed(config["seed"]))
    history, epochs, cursor, best, initial = [], [], [0, 0], float("inf"), None
    if args.resume:
        load_trainable_state(model, saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        history, epochs = saved["history"], saved["epochs"]
        cursor, best, initial = saved["cursor"], saved["best_monitor_loss"], saved["validation_before"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        del saved
        print(f"Resumed {len(history)} updates; next epoch={cursor[0]+1}, batch={cursor[1]+1}", flush=True)
    else:
        initial = validation_loss(model, monitor_loader, device, config["precision"])
        print("Initial monitor loss:", initial, flush=True)
    write_json(out / "validation_before.json", initial)
    # Reconcile logs with the authoritative checkpoint after an interrupted session.
    write_csv(out / "train_metrics.csv", history)
    write_csv(out / "epoch_metrics.csv", epochs)
    torch.cuda.reset_peak_memory_stats()

    def save():
        atomic_save({"model": model_state(model), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "history": history, "epochs": epochs, "cursor": cursor,
            "best_monitor_loss": best, "validation_before": initial, "config": config,
            "fingerprint": fingerprint, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state()}, out / "last.pt")
        write_csv(out / "train_metrics.csv", history)
        write_csv(out / "epoch_metrics.csv", epochs)
        write_json(out / "status.json", {"status": "training", "checkpoint_step": len(history),
                   "completed_epochs": len(epochs), "planned_epochs": config["epochs"]})
        print(f"Saved resumable checkpoint at step {len(history)}", flush=True)

    if not args.resume:
        save()
    stop_epoch = args.stop_after_epoch or config["epochs"]
    start_epoch, start_batch = cursor
    for epoch in range(start_epoch, stop_epoch):
        batches, dropped = plans[epoch]
        offset = start_batch if epoch == start_epoch else 0
        loader = DataLoader(train, batch_sampler=batches[offset:], collate_fn=collate,
                            num_workers=0, generator=torch.Generator().manual_seed(config["seed"] + epoch))
        for batch_index, batch in enumerate(loader, offset):
            started = time.monotonic()
            lr = optimizer.param_groups[0]["lr"]
            def overflow(event):
                event.update(epoch=epoch+1, batch=batch_index+1, next_step=len(history)+1)
                with (out / "amp_overflows.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event) + "\n")
                print(json.dumps({"overflow": event}), flush=True)
            metrics = train_batch(model, batch, optimizer, scaler, config, device, overflow)
            scheduler.step()  # Advance only after a finite optimizer update.
            record = {"epoch": epoch + 1, "batch": batch_index + 1, "step": len(history) + 1,
                      "samples": len(batch["text_input"]), "lr": lr,
                      **metrics, "seconds": time.monotonic() - started}
            history.append(record)
            cursor = [epoch, batch_index + 1]
            with (out / "train_metrics.csv").open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(record))
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(record)
            if record["step"] % 25 == 0 or batch_index == offset:
                print(json.dumps(record), flush=True)
            if record["step"] % config["checkpoint_every_steps"] == 0:
                save()
        # Validation also runs if resuming immediately after the final batch.
        val = validation_loss(model, monitor_loader, device, config["precision"])
        rows = [row for row in history if row["epoch"] == epoch + 1]
        count = sum(row["samples"] for row in rows)
        if count != len(batches) * config["batch_size"]:
            raise RuntimeError("Epoch coverage does not match the batch plan")
        epoch_row = {"epoch": epoch + 1, "step": len(history), "train_samples": count,
                     "dropped_samples": len(dropped), "monitor_samples": len(monitor),
                     **{f"train_{key}": sum(r[key]*r["samples"] for r in rows)/count
                        for key in ("loss", "loss_itc", "loss_itm", "loss_ttc")},
                     "monitor_loss": sum(val.values()),
                     **{f"monitor_{k}": v for k, v in val.items()}}
        epochs.append(epoch_row)
        cursor = [epoch + 1, 0]
        if epoch_row["monitor_loss"] < best:
            best = epoch_row["monitor_loss"]
            atomic_save({"model": model_state(model), "config": config,
                         "epoch": epoch + 1, "monitor_loss": best,
                         "selection": "fixed monitor loss, NOT Recall@K"}, out / "best_monitor.pt")
        print(json.dumps(epoch_row), flush=True)
        save()
        # Curves and a Vietnamese report are refreshed after every completed epoch.
        from scripts.report_unifashion_finetune import generate
        generate(out)
    complete = len(epochs) == config["epochs"]
    write_json(out / "status.json", {"status": "completed" if complete else "session_complete_training_pending",
               "completed_epochs": len(epochs), "planned_epochs": config["epochs"],
               "optimizer_steps": len(history), "checkpoint_step": len(history),
               "precision": config["precision"], "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
               "best_monitor_loss": best, "retrieval_evaluated": False})
    from scripts.report_unifashion_finetune import generate
    generate(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/unifashion_finetune_colab.json")
    p.add_argument("--upstream", type=Path, default=ROOT / "third_party/UniFashion")
    p.add_argument("--assets", type=Path, default=ROOT / "checkpoints/unifashion")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--stop-after-epoch", type=int,
                   help="Session boundary only; does not shorten the configured LR schedule")
    args = p.parse_args()
    if args.resume:
        if args.resume.resolve() != (args.output / "last.pt").resolve() or not args.resume.is_file():
            p.error("Resume must use this output directory's existing last.pt")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "run.log").open("a", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                execute(args)
            except Exception as error:
                status_path = args.output / "status.json"
                status = json.loads(status_path.read_text()) if status_path.exists() else {}
                status.update(status="failed", error=str(error),
                              resume_available=(args.output / "last.pt").exists())
                write_json(status_path, status)
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
