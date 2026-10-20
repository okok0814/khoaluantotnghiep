"""Controlled-study and resume tests; no GPU thesis results are manufactured."""
import copy
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from src.fashioniq_evaluation import digest, protocol
from src.unifashion_training import model_state
from src.unifashion_tuning import balanced_train, prepare_inputs, progress, trial_batches, verify_study
from scripts.report_unifashion_tuning import checked_result, choose, generate
from scripts.tune_unifashion import run_updates, trial_config


def test_same_examples_order_and_negative_uniqueness_across_batches():
    data = pd.DataFrame([{"category": c, "target": f"{c}{i // 2}"}
                         for c in ("dress", "shirt", "toptee") for i in range(20)])
    selected = balanced_train(data, 4, 42)
    assert selected.target.is_unique and len(selected) == 12
    pd.testing.assert_frame_equal(selected, balanced_train(data, 4, 42))
    a, b = trial_batches(12, 2), trial_batches(12, 4)
    assert [i for batch in a for i in batch] == [i for batch in b for i in batch] == list(range(12))
    with pytest.raises(ValueError):
        trial_batches(11, 4)


def inputs(tmp_path):
    data, evaluation = tmp_path / "data", tmp_path / "evaluation"
    data.mkdir()
    evaluation.mkdir()
    rows = []
    captions = {}
    for c in ("dress", "shirt", "toptee"):
        names = [f"{c}{i}" for i in range(100)]
        (evaluation / f"split.{c}.val.json").write_text(json.dumps(names))
        captions[c] = {name: "fixture caption" for name in names}
        rows += [{"category": c, "candidate": names[i], "target": names[i + 20], "text": "red"} for i in range(10)]
    pd.DataFrame(rows).to_csv(evaluation / "queries.csv", index=False)
    (evaluation / "gallery_captions.json").write_text(json.dumps(captions))
    pd.DataFrame([{"category": c, "candidate": f"train_{c}{i}", "target": f"target_{c}{i}"}
                  for c in captions for i in range(8)]).to_csv(data / "train.csv", index=False)
    (data / "data_audit.json").write_text(json.dumps({"files": {"train.csv": digest(data / "train.csv")}}))
    config = {"seed": 42, "precision": "fp32", "scheduler": "constant", "train_per_category": 4,
              "screen_queries_per_category": 4, "screen_gallery_per_category": 60,
              "checkpoint_every_updates": 2, "selection_method": "unifashion_image_caption_score",
              "selection_metric": "mean_macro_recall_10_50",
              "trials": [{"name": "a", "learning_rate": 1e-5, "batch_size": 2}]}
    return data, evaluation, config


def test_screen_is_fixed_complete_distinct_from_full_and_tamper_checked(tmp_path):
    data, evaluation, config = inputs(tmp_path)
    for name in ("one", "two"):
        prepare_inputs(data, evaluation, tmp_path / name, config)
    one, two = verify_study(tmp_path / "one"), verify_study(tmp_path / "two")
    assert one == two
    assert one["screen_protocol"] != one["source_protocol"]
    queries, galleries, _ = protocol(tmp_path / "one/screen/evaluation_data")
    assert len(queries) == 12 and all(len(g) == 60 for g in galleries.values())
    (tmp_path / "one/train.csv").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        verify_study(tmp_path / "one")


def test_trial_resume_restores_rng_optimizer_and_full_equal_budget(tmp_path, monkeypatch):
    import scripts.tune_unifashion as runner

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual_encoder = nn.Linear(2, 2).requires_grad_(False)
            self.projection = nn.Linear(2, 2)
            self.dropout = nn.Dropout(.2)

        def forward(self, batch):
            loss = (self.projection(self.dropout(batch["x"])) - 1).square().mean()
            return {"loss_itc": loss, "loss_itm": loss * .3, "loss_ttc": loss * .2}

    monkeypatch.setattr(runner, "collate", lambda rows: {"x": torch.stack(rows), "text_input": ["x"] * len(rows)})
    torch.manual_seed(42)
    model = Tiny()
    other = copy.deepcopy(model)
    cfg = {"precision": "fp32", "seed": 42, "learning_rate": 1e-5, "betas": [.9, .98],
           "eps": 1e-7, "weight_decay": .05, "max_grad_norm": 1., "max_amp_retries": 2}
    dataset = [torch.tensor([i + 1., i + 2.]) for i in range(12)]
    plan = trial_batches(12, 2)
    metadata = {"checkpoint_every_updates": 2}
    for name in ("continuous", "resumed"):
        (tmp_path / name).mkdir()
    rng = torch.get_rng_state()
    run_updates(model, dataset, cfg, tmp_path / "continuous", metadata, {}, plan, device=torch.device("cpu"))
    torch.set_rng_state(rng)
    run_updates(other, dataset, cfg, tmp_path / "resumed", metadata, {}, plan, device=torch.device("cpu"), stop_after=2)
    saved = torch.load(tmp_path / "resumed/trial.pt", weights_only=True)
    assert not saved["tuning"]["trial_completed"]
    run_updates(other, dataset, cfg, tmp_path / "resumed", metadata, {}, plan,
                saved=saved, device=torch.device("cpu"))
    for key, value in model_state(model).items():
        torch.testing.assert_close(value, model_state(other)[key], rtol=0, atol=0)
    end = torch.load(tmp_path / "resumed/trial.pt", weights_only=True)
    assert len(end["history"]) == 6 and end["tuning"]["trial_examples"] == 12
    assert end["tuning"]["trial_completed"] and progress(end)["completed_epochs"] is None
    assert len(pd.read_csv(tmp_path / "resumed/train_metrics.csv")) == 6


def test_trial_config_keeps_parent_unchanged_and_rejects_precision_change():
    parent = {"precision": "fp32", "learning_rate": 2e-5, "batch_size": 2, "scheduler": "one_cycle"}
    result = trial_config(parent, {"learning_rate": 5e-6, "batch_size": 4})
    assert parent["scheduler"] == "one_cycle" and parent["batch_size"] == 2
    assert result["scheduler"] == "constant" and result["batch_size"] == 4
    with pytest.raises(ValueError, match="FP32"):
        trial_config({**parent, "precision": "fp16"}, {"learning_rate": 1e-5, "batch_size": 4})


def test_selection_retains_parent_and_rejects_mixed_protocol_or_parent(tmp_path):
    assert choose([{"name": "base", "mean_R": 50}, {"name": "a", "mean_R": 50}]) == "base"
    assert choose([{"name": "base", "mean_R": 50}, {"name": "a", "mean_R": 51}]) == "a"
    result = {"protocol": {"hash": "screen"}, "provenance": {"checkpoint_sha256": "parent"},
              "summary": [{"category": "MACRO", "Recall@10": 40., "Recall@50": 60.}]}
    (tmp_path / "method.json").write_text(json.dumps(result))
    (tmp_path / "status.json").write_text(json.dumps({"status": "evaluation_completed", "checkpoint_sha256": "parent"}))
    with pytest.raises(ValueError, match="mismatched"):
        checked_result(tmp_path, "method", {"hash": "full"}, "parent")
    with pytest.raises(ValueError, match="parent"):
        checked_result(tmp_path, "method", {"hash": "screen"}, "other")


def test_report_pending_never_invents_selection(tmp_path):
    data, evaluation, config = inputs(tmp_path)
    root = tmp_path / "study"
    prepare_inputs(data, evaluation, root / "inputs", config)
    (root / "parent.json").write_text(json.dumps({"checkpoint_sha256": "fixture", "progress": {"optimizer_steps": 60000}}))
    selection = generate(root)
    assert not selection["ready"] and selection["selected"] is None
    report = (root / "report/weekly_report.md").read_text(encoding="utf-8")
    assert "Chưa có kết quả" in report


def test_report_full_confirmation_requires_same_parent_and_study(tmp_path):
    data, evaluation, config = inputs(tmp_path)
    root = tmp_path / "study"
    manifest = prepare_inputs(data, evaluation, root / "inputs", config)
    (root / "parent.json").write_text(json.dumps({"checkpoint_sha256": "parent", "progress": {"optimizer_steps": 60000}}))
    trial = config["trials"][0]
    out = root / "trials/a"
    out.mkdir(parents=True)
    (out / "status.json").write_text(json.dumps({"status": "trial_completed"}))
    tuning = {"trial": trial, "parent_sha256": "parent", "trial_completed": True,
              "trial_examples": manifest["train_examples"], "study_sha256": digest(root / "inputs/study.json")}
    for stage in ("screen", "full"):
        for name, score in (("base", 50), ("a", 52)):
            path = root / stage / name
            path.mkdir(parents=True)
            sha = "parent" if name == "base" else "trial"
            p = {"checkpoint_sha256": sha, "evaluation_scope":
                 "screening_only_reduced_gallery" if stage == "screen" else "full_validation"}
            if name == "a":
                p["tuning"] = tuning
            result = {"protocol": manifest["screen_protocol" if stage == "screen" else "source_protocol"],
                      "provenance": p, "summary": [{"category": "MACRO", "Recall@10": score, "Recall@50": score + 20}]}
            (path / "unifashion_image_caption_score.json").write_text(json.dumps(result))
            (path / "status.json").write_text(json.dumps({"status": "evaluation_completed", "checkpoint_sha256": sha}))
    selection = generate(root)
    assert selection["ready"] and selection["selected"] == "a"
    assert "+2.00 pp" in (root / "report/weekly_report.md").read_text(encoding="utf-8")
    path = root / "full/a/unifashion_image_caption_score.json"
    invalid = json.loads(path.read_text())
    invalid["provenance"]["tuning"]["study_sha256"] = "different"
    path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="different training"):
        generate(root)
