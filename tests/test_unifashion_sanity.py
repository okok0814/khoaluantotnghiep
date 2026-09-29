"""Regression checks for real data contracts and upstream training integration.

The tiny model below is a unit-test fixture, not thesis experiment evidence.
"""
import json
from pathlib import Path

import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn

from scripts.prepare_unifashion_sanity import caption_lookup
from src.unifashion_runtime import import_retrieval_class, load_domain_weights, load_vision_weights
from src.unifashion_sanity_data import UniFashionSanityDataset, collate

ROOT = Path(__file__).resolve().parents[1]


def test_qformer_import_without_legacy_transformers_reexports(monkeypatch):
    from src.unifashion_runtime import _import_qformer
    import transformers.modeling_utils as modeling
    import transformers.pytorch_utils as helpers
    root = ROOT / "third_party/UniFashion/src/lavis"
    if not root.exists():
        pytest.skip("Run scripts/download_unifashion.py first")
    for name in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
        monkeypatch.delattr(modeling, name, raising=False)
    # Recreate exactly the missing re-exports in the reported Colab failure.
    module = _import_qformer(root)
    assert module.apply_chunking_to_forward is helpers.apply_chunking_to_forward
    assert module.find_pruneable_heads_and_indices is helpers.find_pruneable_heads_and_indices
    assert module.prune_linear_layer is helpers.prune_linear_layer
    assert module.PreTrainedModel is modeling.PreTrainedModel


def test_caption_lookup_uses_first_released_caption(tmp_path):
    path = tmp_path / "captions.json"
    path.write_text(json.dumps([{"image": "a", "caption": "one"},
                                {"image": "a", "caption": "two"}]))
    lookup, conflicts = caption_lookup(path)
    assert lookup == {"a": "one"}
    assert conflicts == 1


def test_caption_enriched_csv_and_batch_contract(tmp_path):
    rows = []
    for i in range(4):
        Image.new("RGB", (40, 25), (i * 50, 30, 70)).save(tmp_path / f"{i}.jpg")
    for i in (0, 2):
        rows.append({"candidate": str(i), "target": str(i + 1), "modifier": "Make it red!",
                     "category": "dress", "reference_caption": "Blue dress.", "target_caption": "Red dress."})
    path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    dataset = UniFashionSanityDataset(path, tmp_path, {"image_size": 224, "target_ratio": 1.25})
    batch = collate([dataset[0], dataset[1]])
    assert batch["image"].shape == batch["target"].shape == (2, 3, 224, 224)
    assert batch["text_input"] == ["make it red", "make it red"]
    assert batch["target_caption"] == ["red dress", "red dress"]
    rows[1]["target_caption"] = ""
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match="target_caption"):
        UniFashionSanityDataset(path, tmp_path, {"image_size": 224, "target_ratio": 1.25})


def test_official_forward_backward_and_reload_with_tiny_fixture(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    upstream = ROOT / "third_party/UniFashion"
    if not upstream.exists():
        pytest.skip("Run scripts/download_unifashion.py first")
    official = import_retrieval_class(upstream)
    from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
    from transformers import BertTokenizer

    class TinyVision(nn.Module):
        num_features = 16

        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(3, 16, 4, stride=4)

        def forward(self, x):
            return self.proj(x).flatten(2).transpose(1, 2)

    class TinyModel(official):
        @classmethod
        def init_tokenizer(cls):
            tokenizer = BertTokenizer.from_pretrained(str(ROOT / "checkpoints/unifashion/bert-base-uncased"), local_files_only=True)
            tokenizer.add_special_tokens({"bos_token": "[DEC]"})
            return tokenizer

        def init_vision_encoder(self, *args):
            return TinyVision(), nn.LayerNorm(16)

        @classmethod
        def init_Qformer(cls, n, width, freq):
            cfg = BertConfig(vocab_size=30523, hidden_size=32, num_hidden_layers=2,
                             num_attention_heads=4, intermediate_size=64,
                             encoder_width=width, add_cross_attention=True,
                             cross_attention_freq=2, query_length=n)
            return BertLMHeadModel(cfg), nn.Parameter(torch.randn(1, n, 32) * 0.02)

    model = TinyModel(embed_dim=16, max_txt_len=12)
    # Confirm that the forward function is literally inherited, not replaced.
    assert TinyModel.forward is official.forward
    batch = {"image": torch.randn(2, 3, 16, 16), "target": torch.randn(2, 3, 16, 16),
             "text_input": ["make it red", "longer sleeves"],
             "reference_caption": ["blue dress", "white shirt"],
             "target_caption": ["red dress", "shirt with long sleeves"]}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)
    original = model.query_tokens.detach().clone()
    losses = model(batch)
    assert set(losses) == {"loss_itc", "loss_itm", "loss_ttc"}
    loss = sum(losses.values())
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.query_tokens.grad).all()
    assert model.query_tokens.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.visual_encoder.parameters())
    optimizer.step()
    assert not torch.equal(original, model.query_tokens)
    model.eval()
    with torch.no_grad():
        expected = sum(model(batch).values())
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    with torch.no_grad():
        model.query_tokens.add_(1)
    model.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(sum(model(batch).values()), expected)


def test_checkpoint_transfer_rejects_unexplained_keys_and_shapes():
    model = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_domain_weights(model, {"weight": torch.zeros(3, 2)})
    with pytest.raises(ValueError, match="Unaccounted pretrained keys"):
        load_domain_weights(model, {**model.state_dict(), "unknown.weight": torch.zeros(1)})
    with pytest.raises(ValueError, match="do not cover"):
        load_domain_weights(model, {"weight": model.weight})


def test_eva_40_block_checkpoint_loads_complete_39_block_meta_encoder():
    from accelerate import init_empty_weights
    from functools import partial
    upstream = ROOT / "third_party/UniFashion"
    if not upstream.exists():
        pytest.skip("Run scripts/download_unifashion.py first")
    import_retrieval_class(upstream)
    from lavis.models.eva_vit import VisionTransformer

    # Real upstream architecture, with tiny widths/images to avoid allocating GB.
    kwargs = dict(img_size=14, patch_size=14, embed_dim=16, num_heads=2,
                  mlp_ratio=2, qkv_bias=True, use_mean_pooling=False,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
    checkpoint_model = VisionTransformer(depth=40, **kwargs)
    checkpoint_model.norm = nn.LayerNorm(16)
    checkpoint_model.head = nn.Linear(16, 3)
    weights = checkpoint_model.state_dict()
    reference = VisionTransformer(depth=39, **kwargs)
    upstream_result = reference.load_state_dict(weights, strict=False)
    with init_empty_weights():
        actual = VisionTransformer(depth=39, **kwargs)
    assert all(p.is_meta for p in actual.parameters())
    report = load_vision_weights(actual, weights)
    assert report["missing_keys"] == []
    assert report["ignored_checkpoint_keys"] == sorted(upstream_result.unexpected_keys)
    assert len(report["ignored_checkpoint_keys"]) == 17
    assert not any(p.is_meta for p in actual.parameters())
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value)
    actual.eval()
    reference.eval()
    image = torch.randn(2, 3, 14, 14)
    with torch.no_grad():
        torch.testing.assert_close(actual(image), reference(image))


@pytest.mark.parametrize("error", ["missing", "unexpected", "shapes"])
def test_eva_loader_rejects_other_checkpoint_errors_before_assignment(error):
    # A missing required tensor must never be ignored for a meta-initialized model.
    with torch.device("meta"):
        model = nn.Linear(2, 2)
    weights = {"weight": torch.zeros(2, 2), "bias": torch.zeros(2)}
    if error == "missing":
        del weights["bias"]
    elif error == "unexpected":
        weights["blocks.40.norm1.weight"] = torch.zeros(2)
    else:
        weights["weight"] = torch.zeros(3, 2)
    with pytest.raises(ValueError, match="EVA checkpoint mismatch"):
        load_vision_weights(model, weights)
    assert all(p.is_meta for p in model.parameters())
