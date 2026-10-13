"""Protocol edge cases and integration with official UniFashion inference."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from src.fashioniq_evaluation import rank_batch, summarize, normalize


def test_reference_exclusion_rank_boundaries_ties_and_single_query():
    gallery = [f"i{x}" for x in range(61)]
    scores = np.arange(61, 0, -1, dtype=np.float32)[None]
    rows = pd.DataFrame([{"category": "dress", "candidate": "i0", "target": "i10", "text": "red"}])
    row = rank_batch(scores, rows, gallery)[0]
    assert row["target_rank"] == 10 and row["hit@10"] == 1
    assert "i0" not in json.loads(row["top50"])
    rows.loc[0, "target"] = "i50"
    assert rank_batch(scores, rows, gallery)[0]["hit@50"] == 1
    rows.loc[0, "target"] = "i51"
    assert rank_batch(scores, rows, gallery)[0]["hit@50"] == 0
    rows.loc[0, "target"] = "i2"
    tied = rank_batch(np.ones_like(scores), rows, gallery)[0]
    assert tied["target_rank"] == 2
    scores[0, 4] = np.nan
    with pytest.raises(ValueError, match="score"):
        rank_batch(scores, rows, gallery)


def test_macro_micro_are_different_and_metrics_are_percentages():
    rows = [{"category": c, "hit@10": hit, "hit@50": 1}
            for c, hit in [("dress", 1), ("shirt", 0), ("toptee", 0), ("toptee", 0)]]
    result = summarize(rows, {c: ["a", "b"] for c in ("dress", "shirt", "toptee")})
    assert result[-2]["Recall@10"] == pytest.approx(100 / 3)
    assert result[-1]["Recall@10"] == 25
    assert result[-1]["Recall@50"] == 100
    with pytest.raises(ValueError):
        normalize(np.zeros((1, 3)))


def test_official_inference_single_query_matches_batch_and_caption_score():
    from src.unifashion_runtime import import_retrieval_class
    from scripts.evaluate_unifashion_fashioniq import score_queries
    root = Path(__file__).resolve().parents[1]
    if not (root / "third_party/UniFashion").exists():
        pytest.skip("Download upstream assets first")
    official = import_retrieval_class(root / "third_party/UniFashion")
    from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
    from transformers import BertTokenizer
    class Vision(nn.Module):
        num_features = 16
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(3, 16, 4, stride=4)
        def forward(self, x):
            return self.proj(x).flatten(2).transpose(1, 2)
    class Tiny(official):
        @classmethod
        def init_tokenizer(cls):
            t = BertTokenizer.from_pretrained(str(root / "checkpoints/unifashion/bert-base-uncased"), local_files_only=True)
            t.add_special_tokens({"bos_token": "[DEC]"})
            return t
        def init_vision_encoder(self, *args):
            return Vision(), nn.LayerNorm(16)
        @classmethod
        def init_Qformer(cls, n, width, freq):
            cfg = BertConfig(vocab_size=30523, hidden_size=32, num_hidden_layers=2,
                num_attention_heads=4, intermediate_size=64, encoder_width=width,
                add_cross_attention=True, cross_attention_freq=2, query_length=n)
            return BertLMHeadModel(cfg), nn.Parameter(torch.randn(1, n, 32) * .02)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = Tiny(embed_dim=16, max_txt_len=12).eval()
    assert Tiny.inference is official.inference
    assert Tiny.extract_target_features is official.extract_target_features
    images = torch.randn(3, 3, 16, 16)
    with torch.no_grad():
        features, _ = model.extract_target_features(images)
        captions = model.extract_caption_features(["red dress", "blue shirt", "white top"])
        both, visual = score_queries(model, images[:2], features, ["red dress", "blue shirt"], captions)
        one, one_visual = score_queries(model, images[:1], features, ["red dress"], captions)
        assert one.shape == one_visual.shape == (1, 3)
        torch.testing.assert_close(one, both[:1], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(one_visual, visual[:1], atol=1e-6, rtol=1e-5)
        zero, zero_visual = score_queries(model, images[:1], features, ["red dress"], torch.zeros_like(captions))
        torch.testing.assert_close(zero, zero_visual)


def test_report_refuses_different_protocols(tmp_path):
    from scripts.report_fashioniq_evaluation import generate
    b, e = tmp_path / "baseline", tmp_path / "eval"
    b.mkdir(); e.mkdir()
    (b / "fashionclip_additive.json").write_text(json.dumps({"protocol": {"queries": "a"}}))
    (e / "unifashion_image_score.json").write_text(json.dumps({"protocol": {"queries": "b"}}))
    with pytest.raises(ValueError, match="different"):
        generate(b, e, tmp_path / "report")


def test_report_completed_evaluation_records_progress_and_percentage_points(tmp_path):
    from scripts.report_fashioniq_evaluation import generate
    b, e = tmp_path / "baseline", tmp_path / "eval"
    b.mkdir()
    e.mkdir()
    summaries = [{"category": c, "queries": 2, "gallery_size": 60,
                  "Recall@10": 20., "Recall@50": 40.} for c in ("dress", "shirt", "toptee", "MACRO", "MICRO")]
    baseline = {"method": "fashionclip_additive", "protocol": {"fixture": True}, "summary": summaries}
    (b / "fashionclip_additive.json").write_text(json.dumps(baseline))
    fine = {"method": "unifashion_image_score", "protocol": baseline["protocol"],
            "summary": [{**r, "Recall@10": 25.} for r in summaries],
            "provenance": {"snapshot": "fixture.pt", "checkpoint_sha256": "fixture",
                "optimizer_steps": 29500, "completed_epochs": 3, "completed_batches_in_current_epoch": 2500}}
    (e / "unifashion_image_score.json").write_text(json.dumps(fine))
    output = tmp_path / "report"
    table = generate(b, e, output)
    assert (table[table.method == "unifashion_image_score"]["delta_R@10_pp_vs_additive"] == 5).all()
    report = (output / "evaluation_report.md").read_text(encoding="utf-8")
    assert "29500 update" in report and "3 epoch" in report and "2500 batch" in report
    assert "Chưa có kết quả Recall UniFashion" not in report
    assert (output / "recall_comparison.png").exists()
