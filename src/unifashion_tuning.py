"""Small, controlled warm-start trials; these are not resumes of the full run."""
import json
from pathlib import Path
import random

import pandas as pd

from src.fashioniq_evaluation import CATEGORIES, digest, protocol


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def balanced_train(frame, per_category, seed):
    """Use the same examples/order for every LR and batch size; unique targets."""
    rng = random.Random(seed)
    selected, seen = [], set()
    for category in CATEGORIES:
        indices = frame.index[frame.category == category].tolist()
        rng.shuffle(indices)
        chosen = []
        for index in indices:
            target = frame.loc[index, "target"]
            if target not in seen:
                chosen.append(index)
                seen.add(target)
            if len(chosen) == per_category:
                break
        if len(chosen) != per_category:
            raise ValueError(f"Not enough distinct targets in {category}")
        selected.extend(chosen)
    rng.shuffle(selected)
    return frame.loc[selected].reset_index(drop=True)


def trial_batches(size, batch_size):
    if batch_size < 2 or size % batch_size:
        raise ValueError("Need batch >= 2 and exact equal-example coverage")
    return [list(range(i, i + batch_size)) for i in range(0, size, batch_size)]


def progress(saved):
    """Never present short-trial updates as additional full-data epochs."""
    if saved.get("tuning"):
        return {"checkpoint_kind": "tuning_trial", "optimizer_steps": len(saved["history"]),
                "completed_epochs": None, "next_epoch_one_based": None,
                "completed_batches_in_current_epoch": None, "planned_epochs": None,
                "tuning": saved["tuning"]}
    return {"checkpoint_kind": "full_training", "optimizer_steps": len(saved["history"]),
            "completed_epochs": len(saved["epochs"]), "next_epoch_one_based": saved["cursor"][0] + 1,
            "completed_batches_in_current_epoch": saved["cursor"][1],
            "planned_epochs": saved["config"]["epochs"]}


def prepare_inputs(data_dir, eval_data, output, config):
    data_dir, eval_data, output = map(Path, (data_dir, eval_data, output))
    if config["precision"] != "fp32" or config["scheduler"] != "constant":
        raise ValueError("This controlled study fixes FP32 and constant LR")
    audit = json.loads((data_dir / "data_audit.json").read_text())
    if digest(data_dir / "train.csv") != audit["files"]["train.csv"]:
        raise ValueError("Training CSV differs from audit")
    queries, galleries, fingerprint = protocol(eval_data)
    source = pd.read_csv(data_dir / "train.csv", keep_default_na=False)
    train = balanced_train(source, config["train_per_category"], config["seed"])
    for trial in config["trials"]:
        trial_batches(len(train), trial["batch_size"])
        if trial["learning_rate"] <= 0:
            raise ValueError("Learning rate must be positive")
    if len({t["name"] for t in config["trials"]}) != len(config["trials"]):
        raise ValueError("Duplicate trial names")
    # Keep reference and target images of screening queries outside all pooled train images.
    train_ids = set(source.candidate) | set(source.target)
    eligible = queries[~queries.candidate.isin(train_ids) & ~queries.target.isin(train_ids)]
    captions = json.loads((eval_data / "gallery_captions.json").read_text())
    rng = random.Random(config["seed"])
    screen_rows, screen_galleries = [], {}
    for category in CATEGORIES:
        pool = eligible[eligible.category == category]
        count = config["screen_queries_per_category"]
        if len(pool) < count:
            raise ValueError(f"Not enough train-disjoint screen queries in {category}")
        rows = pool.sample(n=count, random_state=config["seed"])
        screen_rows.append(rows)
        required = set(rows.candidate) | set(rows.target)
        remaining = [x for x in galleries[category] if x not in required]
        rng.shuffle(remaining)
        n = config["screen_gallery_per_category"] - len(required)
        if n < 0 or n > len(remaining) or config["screen_gallery_per_category"] <= 50:
            raise ValueError("Screen gallery too small/large")
        chosen = required | set(remaining[:n])
        screen_galleries[category] = [x for x in galleries[category] if x in chosen]
    output.mkdir(parents=True, exist_ok=False)
    train.to_csv(output / "train.csv", index=False)
    screen = output / "screen/evaluation_data"
    screen.mkdir(parents=True)
    pd.concat(screen_rows).to_csv(screen / "queries.csv", index=False)
    for category, names in screen_galleries.items():
        write_json(screen / f"split.{category}.val.json", names)
    write_json(screen / "gallery_captions.json",
               {c: {x: captions[c][x] for x in names} for c, names in screen_galleries.items()})
    _, _, screen_protocol = protocol(screen)
    write_json(screen.parent / "preparation.json", {
        "scope": "screening_only_reduced_gallery", "protocol": screen_protocol,
        "files": {p.name: digest(p) for p in screen.iterdir()},
        "warning": "Not full-val Recall. Never compare these scores with Week 9 or the paper."})
    manifest = {"config": config, "training_data_audit_sha256": digest(data_dir / "data_audit.json"),
                "source_protocol": fingerprint, "train_examples": len(train),
                "screen_protocol": screen_protocol,
                "screen_queries": sum(len(r) for r in screen_rows),
                "files": {p.relative_to(output).as_posix(): digest(p)
                          for p in output.rglob("*") if p.is_file()}}
    write_json(output / "study.json", manifest)
    return manifest


def verify_study(study):
    study = Path(study)
    manifest = json.loads((study / "study.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(study / name) != expected:
            raise ValueError(f"Study input changed: {name}")
    return manifest
