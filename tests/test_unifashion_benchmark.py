"""Timing-accounting checks only; these do not establish GPU speed or FP16 stability."""
import pytest
import torch
from scripts.benchmark_unifashion_t4 import measure_updates
import scripts.benchmark_unifashion_t4 as benchmark


def test_timing_excludes_warmup_but_counts_retries_and_examples(monkeypatch):
    ticks = iter([0., 10., 10., 12., 12., 16.])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))
    calls = []
    def update(*args):
        calls.append(1)
        return {"loss": .2, "amp_retries": 1 if len(calls) == 1 else 0}
    monkeypatch.setattr(benchmark, "train_batch", update)
    loader = [{"text_input": ["a", "b", "c", "d"]} for _ in range(3)]
    result = measure_updates(None, loader, None, None, {}, torch.device("cpu"), 1, 2)
    assert result["samples_per_second"] == pytest.approx(8 / 6)
    assert result["seconds_per_update"] == 3
    assert result["amp_retries_including_warmup"] == 1
    assert len(calls) == 3
    assert [r["warmup"] for r in result["updates"]] == [True, False, False]


def test_invalid_benchmark_length_fails_before_training():
    with pytest.raises(ValueError, match="warmup"):
        measure_updates(None, [], None, None, {}, torch.device("cpu"), -1, 2)
    with pytest.raises(ValueError, match="steps"):
        measure_updates(None, [], None, None, {}, torch.device("cpu"), 1, 0)
