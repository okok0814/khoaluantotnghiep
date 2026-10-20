"""Prepare an equal-example study, then train isolated, resumable FP32 trials."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.fashioniq_evaluation import digest
from src.unifashion_sanity_data import UniFashionSanityDataset, collate
from src.unifashion_training import atomic_save, load_trainable_state, model_state
from src.unifashion_tuning import prepare_inputs, progress, trial_batches, verify_study, write_json
from scripts.train_unifashion_finetune import verify_data, write_csv
from scripts.train_unifashion_sanity import Tee, build_model, train_batch


def trial_config(parent_config, trial):
    if parent_config["precision"] != "fp32":
        raise ValueError("This study requires an FP32 parent; do not mix precision with LR/batch changes")
    config = dict(parent_config)
    config.update(learning_rate=trial["learning_rate"], batch_size=trial["batch_size"],
                  scheduler="constant", num_workers=0)
    return config


def run_updates(model, dataset, config, output, metadata, fingerprint, plan, saved=None,
                device=None, stop_after=None):
    """Same loop on GPU or a tiny CPU fixture; restore optimizer and dropout RNG."""
    device = device or torch.device("cuda")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"], betas=tuple(config["betas"]),
        eps=config["eps"], weight_decay=config["weight_decay"], foreach=False)
    scaler = torch.amp.GradScaler(device.type, enabled=False)
    history = []
    if saved is not None:
        load_trainable_state(model, saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"])
    output = Path(output)
    write_csv(output / "train_metrics.csv", history)
    end = len(plan) if stop_after is None else min(stop_after, len(plan))

    def save():
        tuning = {**metadata, "trial_updates": len(history),
                  "trial_examples": sum(row["samples"] for row in history),
                  "trial_completed": len(history) == len(plan)}
        atomic_save({"model": model_state(model), "optimizer": optimizer.state_dict(),
            "history": history, "config": config, "fingerprint": fingerprint, "tuning": tuning,
            "epochs": [], "cursor": [0, len(history)], "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None},
            output / "trial.pt")
        write_csv(output / "train_metrics.csv", history)
        write_json(output / "status.json", {"status": "trial_completed" if tuning["trial_completed"] else "trial_in_progress",
            "checkpoint_updates": len(history), "planned_updates": len(plan), "tuning": tuning,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0})
        print(f"Saved trial checkpoint: {len(history)}/{len(plan)} updates", flush=True)

    if not history:
        save()
    loader = DataLoader(dataset, batch_sampler=plan[len(history):end], collate_fn=collate,
                        num_workers=0, generator=torch.Generator().manual_seed(config["seed"]))
    for batch in loader:
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.monotonic()
        metrics = train_batch(model, batch, optimizer, scaler, config, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        history.append({"update": len(history) + 1, "samples": len(batch["text_input"]),
                        "lr": config["learning_rate"], **metrics, "seconds": time.monotonic() - started})
        if len(history) % 25 == 0 or len(history) == 1:
            print(json.dumps(history[-1]), flush=True)
        if len(history) % metadata["checkpoint_every_updates"] == 0:
            save()
    save()
    return history


def train(args):
    manifest = verify_study(args.study)
    trials = [t for t in manifest["config"]["trials"] if t["name"] == args.trial]
    if len(trials) != 1:
        raise ValueError("Unknown trial")
    if args.parent.name == "last.pt":
        raise ValueError("Use a fixed snapshot, not last.pt from ongoing training")
    if not torch.cuda.is_available():
        raise RuntimeError("Run this trial in the Colab GPU notebook")
    parent = torch.load(args.parent, map_location="cpu", mmap=True, weights_only=True)
    if parent.get("tuning") or not parent.get("history"):
        raise ValueError("Parent must be a nonempty full-training checkpoint")
    config = trial_config(parent["config"], trials[0])
    fingerprint = parent["fingerprint"]
    if digest(args.data_dir / "data_audit.json") != manifest["training_data_audit_sha256"]:
        raise ValueError("Wrong training data")
    if fingerprint["data_audit"] != manifest["training_data_audit_sha256"]:
        raise ValueError("Parent training data differs")
    # Shared original source remains unchanged, so the main training run can still resume.
    for name, sha in fingerprint["code"].items():
        folder = "scripts" if name.startswith("train_") else "src"
        if digest(ROOT / folder / name) != sha:
            raise ValueError(f"Use original training code bundle: {name}")
    if digest(args.assets / "download_manifest.json") != fingerprint["assets_manifest"]:
        raise ValueError("Use original pretrained assets")
    asset_info = json.loads((args.assets / "download_manifest.json").read_text())
    for suffix in ("eva_vit_g.pth", "pretrain/none_lora_0.pt"):
        record = [r for r in asset_info["files"] if r["path"].replace("\\", "/").endswith("/" + suffix)]
        if len(record) != 1 or digest(args.assets / suffix) != record[0]["sha256"]:
            raise ValueError(f"Pretrained weight changed: {suffix}")
    verify_data(args.data_dir, args.images_dir)
    metadata = {"parent_sha256": digest(args.parent), "parent_progress": progress(parent),
        "trial": trials[0], "study_sha256": digest(args.study / "study.json"),
        "planned_examples": manifest["train_examples"], "optimizer_restarted": True,
        "schedule": "constant", "precision": config["precision"],
        "checkpoint_every_updates": manifest["config"]["checkpoint_every_updates"],
        "code_sha256": {p.name: digest(p) for p in (Path(__file__), ROOT / "src/unifashion_tuning.py")}}
    saved = None
    if args.resume:
        saved = torch.load(args.output / "trial.pt", map_location="cpu", mmap=True, weights_only=True)
        if saved["config"] != config or saved["fingerprint"] != fingerprint:
            raise ValueError("Trial config or training fingerprint changed")
        if any(saved["tuning"][k] != v for k, v in metadata.items()):
            raise ValueError("Parent/study/trial code changed; cannot resume")
    dataset = UniFashionSanityDataset(args.study / "train.csv", args.images_dir, config)
    plan = trial_batches(len(dataset), config["batch_size"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(2)
    device = torch.device("cuda")
    model, load_report = build_model(args.upstream, args.assets, config, device)
    load_trainable_state(model, parent["model"])
    del parent
    torch.manual_seed(config["seed"])  # same starting RNG for all trials
    torch.cuda.reset_peak_memory_stats()
    write_json(args.output / "config.json", config)
    write_json(args.output / "checkpoint_load.json", load_report)
    write_json(args.output / "environment.json", {"gpu": torch.cuda.get_device_name(),
               "torch": str(torch.__version__), "python": sys.version, "metadata": metadata})
    run_updates(model, dataset, config, args.output, metadata, fingerprint, plan,
                saved=saved, device=device, stop_after=args.stop_after_updates)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--source", type=Path, required=True)
    snap.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--data-dir", type=Path, required=True)
    prepare.add_argument("--eval-data", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--config", type=Path, default=ROOT / "configs/unifashion_tuning.json")
    t = sub.add_parser("train")
    for name in ("study", "parent", "data-dir", "images-dir", "output"):
        t.add_argument("--" + name, type=Path, required=True)
    t.add_argument("--trial", required=True)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--stop-after-updates", type=int, help="Session boundary; keeps the full example budget")
    t.add_argument("--assets", type=Path, default=ROOT / "checkpoints/unifashion")
    t.add_argument("--upstream", type=Path, default=ROOT / "third_party/UniFashion")
    args = p.parse_args()
    if args.command == "snapshot":
        args.output.mkdir(parents=True, exist_ok=False)
        destination = args.output / "parent_snapshot.pt"
        before = (args.source.stat().st_size, args.source.stat().st_mtime_ns)
        shutil.copyfile(args.source, destination)
        after = (args.source.stat().st_size, args.source.stat().st_mtime_ns)
        if before != after or destination.stat().st_size != before[0]:
            raise RuntimeError("Source changed while copying; stop training and create a new study")
        saved = torch.load(destination, map_location="cpu", mmap=True, weights_only=True)
        if saved.get("tuning") or not saved.get("history") or saved["config"]["precision"] != "fp32":
            raise ValueError("Expected a saved FP32 full-training checkpoint")
        metadata = {"checkpoint_sha256": digest(destination), "progress": progress(saved),
                    "config": saved["config"], "source": str(args.source),
                    "next_lr_in_original_run": saved["optimizer"]["param_groups"][0]["lr"]}
        write_json(args.output / "parent.json", metadata)
        print(json.dumps(metadata, indent=2))
        return
    if args.command == "prepare":
        result = prepare_inputs(args.data_dir, args.eval_data, args.output,
                                json.loads(args.config.read_text()))
        print(json.dumps({k: result[k] for k in ("train_examples", "screen_queries")}, indent=2))
        return
    if args.stop_after_updates is not None and args.stop_after_updates < 1:
        p.error("stop-after-updates must be positive")
    if args.resume:
        if not (args.output / "trial.pt").is_file():
            p.error("No trial checkpoint to resume")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "run.log").open("a", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                train(args)
            except Exception as exc:
                status_path = args.output / "status.json"
                status = json.loads(status_path.read_text()) if status_path.exists() else {}
                status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                write_json(status_path, status)
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
