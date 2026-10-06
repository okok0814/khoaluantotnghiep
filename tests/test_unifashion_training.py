"""CPU regression fixtures, not results from a trained thesis model."""
import copy
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from scripts.train_unifashion_sanity import train_batch
from scripts.train_unifashion_finetune import write_csv
from src.unifashion_training import atomic_save, load_trainable_state, make_batches, model_state


def test_unique_target_batches_preserve_queries_and_are_reproducible():
    targets = ["a"] * 5 + ["b"] * 3 + ["c"] * 4
    batches, dropped = make_batches(targets, 2, 42, 0)
    assert dropped == []
    assert sorted(i for b in batches for i in b) == list(range(len(targets)))
    assert all(len({targets[i] for i in b}) == 2 for b in batches)
    assert (batches, dropped) == make_batches(targets, 2, 42, 0)
    assert batches != make_batches(targets, 2, 42, 1)[0]
    batches, dropped = make_batches(["a", "a", "a", "b"], 2, 42, 0)
    assert len(batches) == 1 and len(dropped) == 2
    assert sorted([i for b in batches for i in b] + dropped) == list(range(4))


class TinyLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_encoder = nn.Linear(2, 2)
        self.visual_encoder.requires_grad_(False)
        self.projection = nn.Linear(2, 2)
        self.dropout = nn.Dropout(.2)

    def forward(self, batch):
        loss = (self.projection(self.dropout(batch["x"])) - 1).square().mean()
        return {"loss_itc": loss, "loss_itm": loss * .3, "loss_ttc": loss * .2}


def setup_training():
    model = TinyLossModel()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5,
                                 betas=(.9, .98), eps=1e-7, weight_decay=.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-5, total_steps=12,
                                                   pct_start=.25, div_factor=100)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return model, optimizer, scheduler, scaler


CONFIG = {"precision": "fp32", "max_amp_retries": 3, "max_grad_norm": 1.0}
BATCH = {"x": torch.tensor([[1., 2.], [3., 4.]])}


def test_resume_matches_uninterrupted_dropout_optimizer_and_schedule(tmp_path):
    torch.manual_seed(42)
    model, opt, schedule, scaler = setup_training()
    expected_metrics = []
    for step in range(12):
        expected_metrics.append(train_batch(model, BATCH, opt, scaler, CONFIG, torch.device("cpu")))
        schedule.step()
        if step == 4:
            atomic_save({"model": model_state(model), "optimizer": opt.state_dict(),
                         "scheduler": schedule.state_dict(), "rng": torch.get_rng_state()}, tmp_path / "last.pt")
    expected = copy.deepcopy(model_state(model))
    restored, opt2, schedule2, scaler2 = setup_training()
    saved = torch.load(tmp_path / "last.pt", weights_only=True, mmap=True)
    load_trainable_state(restored, saved["model"])
    opt2.load_state_dict(saved["optimizer"])
    schedule2.load_state_dict(saved["scheduler"])
    torch.set_rng_state(saved["rng"])
    for step in range(5, 12):
        observed = train_batch(restored, BATCH, opt2, scaler2, CONFIG, torch.device("cpu"))
        assert observed == expected_metrics[step]
        schedule2.step()
    for key, value in model_state(restored).items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    assert schedule2.state_dict() == schedule.state_dict()
    assert not (tmp_path / "last.pt.tmp").exists()
    with pytest.raises(ValueError, match="missing/unexpected"):
        load_trainable_state(restored, {})


def test_amp_overflow_retries_without_corrupting_optimizer():
    torch.manual_seed(12)
    model, optimizer, _, _ = setup_training()
    initial = copy.deepcopy(model)
    initial_rng = torch.get_rng_state()
    scaler = torch.amp.GradScaler("cpu", init_scale=256)
    handle = model.projection.weight.register_hook(
        lambda grad: grad * float("inf") if scaler.get_scale() > 64 else grad)
    events = []
    result = train_batch(model, BATCH, optimizer, scaler, CONFIG, torch.device("cpu"), events.append)
    handle.remove()
    assert result["amp_retries"] == 2
    assert result["loss_scale_after"] == 64
    assert all(not e["optimizer_update_applied"] for e in events)
    # The optimizer must see only one valid update, with the same dropout mask.
    assert all(state["step"].item() == 1 for state in optimizer.state.values())
    reference_opt = torch.optim.AdamW([p for p in initial.parameters() if p.requires_grad],
        lr=optimizer.param_groups[0]["lr"], betas=optimizer.param_groups[0]["betas"],
        eps=1e-7, weight_decay=.05)
    torch.set_rng_state(initial_rng)
    train_batch(initial, BATCH, reference_opt, torch.amp.GradScaler("cpu", enabled=False), CONFIG, torch.device("cpu"))
    for key, value in model_state(model).items():
        torch.testing.assert_close(value, model_state(initial)[key], rtol=0, atol=0)


def test_persistent_nonfinite_gradients_fail_without_update():
    model, optimizer, _, _ = setup_training()
    original = copy.deepcopy(model_state(model))
    model.projection.weight.register_hook(lambda grad: grad * float("inf"))
    with pytest.raises(FloatingPointError, match="persist"):
        train_batch(model, BATCH, optimizer, torch.amp.GradScaler("cpu", init_scale=256),
                    CONFIG, torch.device("cpu"))
    assert len(optimizer.state) == 0
    for key, value in model_state(model).items():
        torch.testing.assert_close(value, original[key])


def test_resume_from_initial_checkpoint_removes_uncommitted_csv(tmp_path):
    path = tmp_path / "train_metrics.csv"
    write_csv(path, [{"step": 1, "loss": .5}])
    write_csv(path, [])
    assert not path.exists()


def test_report_uses_only_checkpointed_rows(tmp_path):
    from scripts.report_unifashion_finetune import generate
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "config.json").write_text((root / "configs/unifashion_finetune_colab.json").read_text())
    (tmp_path / "status.json").write_text(json.dumps({"status": "training", "checkpoint_step": 1}))
    rows = [{"step": i, "loss": .9, "loss_itc": .3, "loss_itm": .3, "loss_ttc": .3,
             "lr": 2e-7, "amp_retries": 0, "grad_norm_before_clip": 2.} for i in (1, 2)]
    pd.DataFrame(rows).to_csv(tmp_path / "train_metrics.csv", index=False)
    report = generate(tmp_path).read_text(encoding="utf-8")
    assert "0/10 epoch" in report and "1 cập nhật" in report
    assert "Chưa có epoch hoàn tất" in report
    assert (tmp_path / "training_curves.png").is_file()


def test_full_runner_epoch_stop_and_resume_matches_continuous_run(tmp_path, monkeypatch):
    """Exercise the real runner on CPU fixtures, replacing only GPU/model/data I/O."""
    from types import SimpleNamespace
    import scripts.train_unifashion_finetune as runner

    class CPUFacade:
        cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda: "CPU unit-test fixture",
            reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 0,
            get_rng_state=torch.get_rng_state, set_rng_state=torch.set_rng_state)

        def device(self, _):
            return torch.device("cpu")

        def __getattr__(self, name):
            return getattr(torch, name)

    class TinyData(torch.utils.data.Dataset):
        def __init__(self, path, *args, **kwargs):
            self.n = 6 if path.name == "train.csv" else 4
            self.data = pd.DataFrame({"target": [str(i) for i in range(self.n)]})

        def __len__(self):
            return self.n

        def __getitem__(self, index):
            return torch.tensor([index + 1., index + 2.])

    def pack(samples):
        x = torch.stack(samples)
        return {"x": x, "image": x, "text_input": ["fixture"] * len(samples)}

    monkeypatch.setattr(runner, "torch", CPUFacade())
    monkeypatch.setattr(runner, "UniFashionSanityDataset", TinyData)
    monkeypatch.setattr(runner, "collate", pack)
    monkeypatch.setattr(runner, "build_model", lambda *args: (TinyLossModel(), {"unit_test_fixture": True}))
    monkeypatch.setattr(runner, "verify_data", lambda *args: {"unit_test_fixture": True})
    real_hash = runner.file_hash
    monkeypatch.setattr(runner, "file_hash", lambda p:
        "f272d3a3cc91ae809f844d3e258f6a872be405cb203a550c15c4510c7280b541"
        if p.name == "none_lora_0.pt" else real_hash(p))
    (tmp_path / "data_audit.json").write_text("{}")
    (tmp_path / "download_manifest.json").write_text("{}")
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/unifashion_finetune_colab.json").read_text())
    config.update(epochs=3, checkpoint_every_steps=2)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    args = SimpleNamespace(output=tmp_path / "continuous", config=config_path,
        assets=tmp_path, data_dir=tmp_path, images_dir=tmp_path, upstream=tmp_path,
        resume=None, stop_after_epoch=None)
    args.output.mkdir()
    runner.execute(args)
    continuous = torch.load(args.output / "last.pt", weights_only=True)
    args.output = tmp_path / "resumed"
    args.output.mkdir()
    args.stop_after_epoch = 1
    runner.execute(args)
    first = json.loads((args.output / "status.json").read_text())
    assert first["completed_epochs"] == 1 and first["planned_epochs"] == 3
    assert first["status"] == "session_complete_training_pending"
    # Simulate CSV rows flushed after the checkpoint, then a lost session.
    with (args.output / "train_metrics.csv").open("a") as f:
        f.write("uncommitted,partial,row\n")
    args.resume = args.output / "last.pt"
    args.stop_after_epoch = 3
    runner.execute(args)
    resumed = torch.load(args.resume, weights_only=True)
    assert resumed["cursor"] == [3, 0]
    assert len(resumed["history"]) == 9 and len(resumed["epochs"]) == 3
    assert len(pd.read_csv(args.output / "train_metrics.csv")) == 9
    assert resumed["scheduler"] == continuous["scheduler"]
    assert resumed["epochs"] == continuous["epochs"]
    for key, value in resumed["model"].items():
        torch.testing.assert_close(value, continuous["model"][key], rtol=0, atol=0)
    status = json.loads((args.output / "status.json").read_text())
    assert status["status"] == "completed" and status["optimizer_steps"] == 9
    config["learning_rate"] *= 2
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Resume config/data/code/assets changed"):
        runner.execute(args)
