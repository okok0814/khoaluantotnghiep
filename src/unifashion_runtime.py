"""Small import/initialization adapter for the pinned official UniFashion code.

The official CIR forward method and all three losses are executed unchanged.
Unrelated eager imports are omitted; legacy Q-former utility imports are
redirected to their defining transformers.pytorch_utils module.
This is the retrieval branch, not the LLM/diffusion generation pipeline.
"""
import ast
import importlib
import os
from pathlib import Path
import sys
import types

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache/huggingface"))

import torch
from torch import nn


def _import_qformer(root):
    """Keep upstream math intact while avoiding deprecated utility re-exports."""
    name = "lavis.models.blip2_models.Qformer"
    path = root / "models/blip2_models/Qformer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    helpers = {"apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "transformers.modeling_utils":
            moved = [alias for alias in node.names if alias.name in helpers]
            node.names = [alias for alias in node.names if alias.name not in helpers]
            if moved:
                body.append(ast.copy_location(ast.ImportFrom(
                    module="transformers.pytorch_utils", names=moved, level=0), node))
            if not node.names:
                continue
        body.append(node)
    tree.body = body
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = "lavis.models.blip2_models"
    sys.modules[name] = module
    try:
        exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
    except Exception:
        del sys.modules[name]
        raise
    return module


def import_retrieval_class(upstream):
    root = Path(upstream).resolve() / "src/lavis"
    module_name = "lavis.models.blip2_models.blip2_qformer_cir_rerank"
    if module_name in sys.modules:
        return sys.modules[module_name].Blip2QformerCirRerank
    # Avoid lavis.__init__, which imports every dataset and generation model.
    for name, relative in (("lavis", ""), ("lavis.common", "common"),
                           ("lavis.models", "models"),
                           ("lavis.models.blip2_models", "models/blip2_models"),
                           ("lavis.models.blip_models", "models/blip_models")):
        if name in sys.modules:
            raise RuntimeError("Use a fresh Python process; another LAVIS package is already imported")
        module = types.ModuleType(name)
        module.__path__ = [str(root / relative)]
        sys.modules[name] = module
    base = importlib.import_module("lavis.models.base_model")
    sys.modules["lavis.models"].BaseModel = base.BaseModel
    registry = importlib.import_module("lavis.common.registry").registry
    registry.register_path("library_root", str(root))
    registry.register_path("cache_root", str(root.parents[2] / ".cache"))
    _import_qformer(root)
    path = root / "models/blip2_models/blip2_qformer_cir_rerank.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and
        (node.module or "").startswith("rq_vae_transformer."))]
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = "lavis.models.blip2_models"
    sys.modules[module_name] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module.Blip2QformerCirRerank


def checkpoint_state(path):
    """Tensor-only load; never enable unrestricted pickle for downloaded weights."""
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as error:
        if "mmap can only be used" not in str(error):
            raise
        raw = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model", "state_dict", "Blip2QformerCirRerank", "Blip2QformerCirFull"):
        if key in raw and isinstance(raw[key], dict):
            raw = raw[key]
            break
    if not raw or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in raw.items()):
        raise ValueError("Expected a tensor state dict")
    return raw


def load_vision_weights(visual, weights):
    """Load EVA's 39-block feature encoder from the released 40-block checkpoint.

    The official create_eva_vit_g uses strict=False: its feature extractor drops
    block 39 (zero-based), the final norm, and the classifier. Account for only
    these known extras, then strictly load every tensor the encoder actually uses.
    """
    unused = {"norm.weight", "norm.bias", "head.weight", "head.bias"}
    unused.update("blocks.39." + suffix for suffix in (
        "norm1.weight", "norm1.bias", "attn.q_bias", "attn.v_bias",
        "attn.qkv.weight", "attn.proj.weight", "attn.proj.bias",
        "norm2.weight", "norm2.bias", "mlp.fc1.weight", "mlp.fc1.bias",
        "mlp.fc2.weight", "mlp.fc2.bias"))
    expected = visual.state_dict()
    missing = sorted(set(expected) - set(weights))
    extras = set(weights) - set(expected)
    unexpected = sorted(extras - unused)
    mismatched = {key: (list(weights[key].shape), list(expected[key].shape))
                  for key in set(expected) & set(weights)
                  if weights[key].shape != expected[key].shape}
    if missing or unexpected or mismatched:
        raise ValueError(f"EVA checkpoint mismatch: missing={missing}, "
                         f"unexpected={unexpected}, shapes={mismatched}")
    visual.load_state_dict({key: weights[key] for key in expected}, strict=True, assign=True)
    if any(tensor.is_meta for tensor in visual.state_dict().values()):
        raise RuntimeError("EVA encoder still contains unmaterialized meta tensors")
    return {"loaded": True, "matched_keys": len(expected),
            "ignored_checkpoint_keys": sorted(extras), "missing_keys": []}


def build_model(upstream, assets, config, device, inspect_only=False):
    from contextlib import nullcontext
    from accelerate import init_empty_weights
    from functools import partial
    from transformers import BertTokenizer

    assets = Path(assets)
    official_class = import_retrieval_class(upstream)
    from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
    from lavis.models.blip2_models.blip2 import LayerNorm
    from lavis.models.eva_vit import VisionTransformer

    class LocalUniFashion(official_class):
        @classmethod
        def init_tokenizer(cls, truncation_side="right"):
            tokenizer = BertTokenizer.from_pretrained(
                str(assets / "bert-base-uncased"), local_files_only=True,
                truncation_side=truncation_side)
            tokenizer.add_special_tokens({"bos_token": "[DEC]"})
            return tokenizer

        @classmethod
        def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
            cfg = BertConfig.from_pretrained(str(assets / "bert-base-uncased"), local_files_only=True)
            cfg.encoder_width = vision_width
            cfg.add_cross_attention = True
            cfg.cross_attention_freq = cross_attention_freq
            cfg.query_length = num_query_token
            model = BertLMHeadModel(cfg)
            query = nn.Parameter(torch.empty(1, num_query_token, cfg.hidden_size))
            nn.init.normal_(query, std=cfg.initializer_range)
            return model, query

        def init_vision_encoder(self, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision):
            # Exact create_eva_vit_g architecture; meta allocation prevents a
            # second 4 GB copy while loading the separately released backbone.
            with init_empty_weights():
                visual = VisionTransformer(
                    img_size=img_size, patch_size=14, use_mean_pooling=False,
                    embed_dim=1408, depth=39, num_heads=16, mlp_ratio=4.3637,
                    qkv_bias=True, drop_path_rate=drop_path_rate,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    use_checkpoint=use_grad_checkpoint)
            if not inspect_only:
                weights = checkpoint_state(assets / "eva_vit_g.pth")
                self.vision_checkpoint_report = load_vision_weights(visual, weights)
                del weights
            else:
                self.vision_checkpoint_report = {"loaded": False, "inspection_only": True}
            # Match frozen visual storage to the selected compute precision.
            # All trainable Q-former parameters stay FP32 for the optimizer.
            if device.type == "cuda" and not inspect_only:
                visual.to(dtype={"fp16": torch.float16, "bf16": torch.bfloat16,
                                 "fp32": torch.float32}[config["precision"]])
            self.vit_name = model_name
            return visual, LayerNorm(visual.num_features)

    with init_empty_weights() if inspect_only else nullcontext():
        model = LocalUniFashion(img_size=config["image_size"], vit_precision="fp16",
                                freeze_vit=True, max_txt_len=config["max_txt_len"])
    state = checkpoint_state(assets / "pretrain/none_lora_0.pt")
    report = load_domain_weights(model, state)
    report["vision_checkpoint"] = model.vision_checkpoint_report
    if not inspect_only:
        model.to(device)
    report["inspection_only"] = inspect_only
    model.visual_encoder.eval()
    return model, report


def load_domain_weights(model, state):
    """Explicitly account for the domain-pretrain -> CIR architecture transfer."""
    expected = model.state_dict()
    matched = {key: value for key, value in state.items() if key in expected}
    mismatch = {key: (list(value.shape), list(expected[key].shape))
                for key, value in matched.items() if value.shape != expected[key].shape}
    if mismatch:
        raise ValueError(f"Checkpoint shape mismatch: {mismatch}")
    result = model.load_state_dict(matched, strict=False, assign=True)
    missing = [key for key in result.missing_keys if not key.startswith("visual_encoder.")]
    # CIR adds learned target prompts and a caption projection to pretraining.
    allowed_missing = {"prompt_tokens", "caption_proj.weight", "caption_proj.bias",
                       "query_proj.weight", "query_proj.bias"}
    # Pretrain does not save the unused BERT language-model head or the
    # deterministic position_ids buffer. CIR forward only calls Qformer.bert.
    allowed_missing.update(key for key in missing if key.startswith("Qformer.cls."))
    allowed_missing.add("Qformer.bert.embeddings.position_ids")
    unexplained = set(missing) - allowed_missing
    if unexplained:
        raise ValueError(f"Pretraining weights do not cover retrieval model: {sorted(unexplained)}")
    extras = sorted(set(state) - set(expected))
    allowed_extras = {"llm_query_tokens", "llm_proj.weight", "llm_proj.bias",
                      "proj_layer.dense1.weight", "proj_layer.dense1.bias",
                      "proj_layer.dense2.weight", "proj_layer.dense2.bias",
                      "proj_layer.LayerNorm.weight", "proj_layer.LayerNorm.bias"}
    if set(extras) - allowed_extras:
        raise ValueError(f"Unaccounted pretrained keys: {sorted(set(extras) - allowed_extras)}")
    report = {"matched_keys": len(matched), "missing_nonvision_keys": missing,
              "pretrained_extra_keys": extras,
              "visual_backbone": "separate official frozen EVA-G checkpoint",
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    return report
