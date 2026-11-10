"""Inference-only adapters; official forward equations, explicit checkpoint variants."""
from contextlib import nullcontext
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import zipfile

import torch
from torch import nn
import torch.nn.functional as F
from src.fashioniq_evaluation import digest
from src.unifashion_sanity_data import clean_caption, preprocess

ROOT = Path(__file__).resolve().parents[1]


def module_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def original_runtime(archive, destination):
    expected = "ab27b7634b40bf271957f67311a775bc7d976184311676dcd957eda875ceed66"
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.endswith("src/unifashion_runtime.py")]
        if len(names) != 1:
            raise ValueError("Original runtime missing/ambiguous")
        content = z.read(names[0])
    import hashlib
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("Original runtime fingerprint differs; do not load the integrated gate for this control")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "original_unifashion_runtime.py"
    path.write_bytes(content)
    return module_file("paper_original_runtime", path)


def tensor_state(path, key=None):
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as error:
        if "mmap can only be used" not in str(error):
            raise
        raw = torch.load(path, map_location="cpu", weights_only=True)
    raw = raw[key] if key else raw
    if not raw or not all(isinstance(k, str) and torch.is_tensor(v) for k,v in raw.items()):
        raise ValueError(f"Not a tensor state dict: {path}")
    return raw


def sprc_class(root, helper, scratch):
    """Import only SPRC's retrieval dependencies, not LAVIS generation/dataset modules."""
    for name, relative in (("lavis", ""), ("lavis.common", "common"), ("lavis.models", "models"),
                           ("lavis.models.blip2_models", "models/blip2_models"),
                           ("lavis.models.blip_models", "models/blip_models")):
        if name in sys.modules:
            raise ValueError("SPRC requires a fresh process, without another LAVIS runtime")
        module = types.ModuleType(name)
        module.__path__ = [str(root / relative)]
        sys.modules[name] = module
    base = importlib.import_module("lavis.models.base_model")
    sys.modules["lavis.models"].BaseModel = base.BaseModel
    registry = importlib.import_module("lavis.common.registry").registry
    registry.register_path("library_root", str(root))
    registry.register_path("cache_root", str(scratch))
    helper._import_qformer(root)
    return importlib.import_module("lavis.models.blip2_models.blip2_qformer_cir_align_prompt").Blip2QformerCirAlignPrompt


def load_sprc_retrieval_weights(model, vision, pretrained, fine):
    """Load every retained tensor; exclude the LM head before loading its resized bias.

    SPRC retrieval calls Qformer.bert directly. Qformer.cls is only used for
    language generation, and its tied decoder/bias can resize inconsistently
    during meta initialization. It must be removed BEFORE load_state_dict,
    since strict=False does not allow a mismatched tensor shape.
    """
    head_prefix = "Qformer.cls."
    ignored = sorted(k for k in set(pretrained) | set(fine) if k.startswith(head_prefix))
    model.Qformer.cls = nn.Identity()
    expected = model.state_dict()
    unexpected = sorted(k for k in fine if k not in expected and not k.startswith(head_prefix))
    if unexpected:
        raise ValueError("Unexpected SPRC weights: " + str(unexpected))
    state = {k: v for k, v in pretrained.items() if k in expected}
    state.update({"visual_encoder."+k: v for k, v in vision.items()
                  if "visual_encoder."+k in expected and "visual_encoder."+k not in state})
    state.update({k: v for k, v in fine.items() if k in expected})
    position_key = "Qformer.bert.embeddings.position_ids"
    missing = sorted(set(expected)-set(state))
    if set(missing)-{position_key}:
        raise ValueError("SPRC active tensors missing: " + str(missing))
    if position_key in missing:
        positions = model.Qformer.bert.embeddings.position_embeddings.num_embeddings
        state[position_key] = torch.arange(positions, dtype=torch.long).expand((1, -1))
    mismatched = {k: {"checkpoint": list(state[k].shape), "model": list(v.shape)}
                  for k, v in expected.items() if state[k].shape != v.shape}
    if mismatched:
        raise ValueError("SPRC retrieval weight shape mismatch: " + str(mismatched))
    model.load_state_dict(state, strict=True, assign=True)
    if any(v.is_meta for v in model.state_dict().values()):
        raise ValueError("SPRC contains uninitialized retrieval tensors")
    return {"matched_keys": len(state), "ignored_language_head_keys": ignored,
            "reconstructed_buffers": missing, "missing_retrieval_keys": []}


class Clip4Cir:
    methods = ("clip4cir_combiner",)
    def __init__(self, assets, **kwargs):
        import clip
        self.clip = clip
        root = assets / "clip4cir"
        records = json.loads((root / "checkpoints.json").read_text())
        states = {}
        for key, outer in (("clip", "CLIP"), ("combiner", "Combiner")):
            path = root / records[key]["path"]
            if digest(path) != records[key]["sha256"]:
                raise ValueError("CLIP4CIR checkpoint changed")
            states[key] = tensor_state(path, outer)
        # build_model infers the architecture from the checkpoint, avoiding another base-weight download.
        from clip.model import build_model
        self.model = build_model(dict(states["clip"])).float().cuda().eval()
        if int(self.model.visual.input_resolution) != 288 or self.model.visual.output_dim != 640:
            raise ValueError("Expected the released RN50x4 fullft variant (288px, 640d)")
        combiner = module_file("paper_clip4cir_combiner", root / "source/src/combiner.py").Combiner
        state = states["combiner"]
        dim = state["image_projection_layer.weight"].shape[1]
        projection = state["image_projection_layer.weight"].shape[0]
        hidden = state["combiner_layer.weight"].shape[0]
        self.combiner = combiner(dim, projection, hidden).float().cuda().eval()
        self.combiner.load_state_dict(state, strict=True)
        self.transform = module_file("paper_clip4cir_data", root / "source/src/data_utils.py").targetpad_transform(1.25, 288)
        self.provenance = {"variant": "FashionIQ/RN50x4_fullft", "checkpoints": records,
                           "precision": "fp32", "extra_gallery_captions": False,
                           "protocol_adaptation": "Shared raw modifier text, audited resized images, reference excluded, stable ties; differs from native published protocol"}

    def gallery(self, images, captions):
        return {"image": self.model.encode_image(images)}

    def scores(self, images, texts, targets):
        image = self.model.encode_image(images)
        text = self.model.encode_text(self.clip.tokenize(texts, truncate=True).cuda())
        query = self.combiner.combine_features(image, text)
        return {self.methods[0]: query @ F.normalize(targets["image"], dim=-1).T}


class Sprc:
    methods = ("sprc_vitl",)
    def __init__(self, assets, original_code, scratch, **kwargs):
        helper = original_runtime(original_code, scratch)
        root = assets / "sprc/source/src/lavis"
        official = sprc_class(root, helper, scratch)
        from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
        from lavis.models.blip2_models.blip2 import LayerNorm
        from lavis.models.clip_vit import VisionTransformer
        from transformers import BertTokenizer
        from accelerate import init_empty_weights
        tokenizer_dir = kwargs["unifashion_assets"] / "bert-base-uncased"

        class LocalSPRC(official):
            @classmethod
            def init_tokenizer(cls, truncation_side="right"):
                tokenizer = BertTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, truncation_side=truncation_side)
                tokenizer.add_special_tokens({"bos_token": "[DEC]"})
                return tokenizer

            @classmethod
            def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
                cfg = BertConfig.from_pretrained(str(tokenizer_dir), local_files_only=True)
                cfg.encoder_width, cfg.add_cross_attention = vision_width, True
                cfg.cross_attention_freq, cfg.query_length = cross_attention_freq, num_query_token
                return BertLMHeadModel(cfg), nn.Parameter(torch.zeros(1, num_query_token, cfg.hidden_size))

            def init_vision_encoder(self, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision):
                self.vit_name = "clip_L"
                return VisionTransformer(img_size, 14, 1024, 23, 16, False), LayerNorm(1024)

        with init_empty_weights():
            model = LocalSPRC(vit_model="clip_L", max_txt_len=32)
        root = assets / "sprc"
        vision = tensor_state(root / "clip_vit_L.pth")
        pretrained = tensor_state(root / "blip2_pretrained_vitL.pth", "model")
        fine = tensor_state(root / "sprc_fiq_vitl.pt", "Blip2QformerCirAlignPrompt")
        load_report = load_sprc_retrieval_weights(model, vision, pretrained, fine)
        self.model = model.float().cuda().eval()
        self.model.maybe_autocast = lambda *a, **kw: nullcontext()
        self.transform = preprocess({"target_ratio": 1.25, "image_size": 224})
        self.provenance = {"variant": "SPRC FashionIQ ViT-L (not the paper's full-model variant)",
                           "precision": "fp32", "retrieval_weight_loading": load_report,
                           "extra_gallery_captions": False, "max_txt_len": 32,
                           "protocol_adaptation": "Shared raw text followed by BLIP cleanup; reference excluded; audited resized images; stable ties"}

    def gallery(self, images, captions):
        features, _ = self.model.extract_target_features(images)
        return {"image": features}

    def scores(self, images, texts, targets):
        reference = self.model.ln_vision(self.model.visual_encoder(images))
        scores = self.model.inference(reference, targets["image"], [clean_caption(t) for t in texts])
        return {self.methods[0]: scores.reshape(len(texts), len(targets["image"]))}


class UniFashionInitial:
    methods = ("unifashion_initial_image_score", "unifashion_initial_image_caption_score")
    def __init__(self, original_code, scratch, unifashion_assets, upstream, reference_result, **kwargs):
        runtime = original_runtime(original_code, scratch)
        reference = json.loads(reference_result.read_text())["provenance"]
        config = reference["training_config"]
        if config["precision"] != "fp32":
            raise ValueError("This control is paired with the FP32 original training run")
        if digest(unifashion_assets / "download_manifest.json") != reference["training_fingerprint"]["assets_manifest"]:
            raise ValueError("Use the original training assets manifest")
        records = json.loads((unifashion_assets / "download_manifest.json").read_text())["files"]
        for rec in records:
            name = rec["path"].replace("\\", "/")
            if name.startswith("third_party/UniFashion/dataset/"):
                continue  # Not read by inference; gallery_captions.json is hashed separately.
            if "third_party/UniFashion/" in name:
                path = upstream / name.split("third_party/UniFashion/", 1)[1]
            elif "checkpoints/unifashion/" in name:
                path = unifashion_assets / name.split("checkpoints/unifashion/", 1)[1]
            else:
                raise ValueError("Unrecognized original asset path: " + name)
            if digest(path) != rec["sha256"]:
                # Reuse the proven LF/CRLF-only restoration, never accept semantic changes.
                from scripts.evaluate_unifashion_final import restore_source_line_endings
                if path.suffix not in (".py", ".md", ".sh", ".yaml"):
                    raise ValueError("Original asset differs: " + str(path))
                restore_source_line_endings(path, rec["sha256"])
        torch.manual_seed(config["seed"])
        self.model, report = runtime.build_model(upstream, unifashion_assets, config, torch.device("cuda"))
        self.model.eval()
        self.model.maybe_autocast = lambda *a, **kw: nullcontext()
        self.transform = preprocess(config)
        self.provenance = {"variant": "domain-pretraining to CIR initialization, BEFORE fine-tuning",
                           "random_retrieval_heads": report["missing_nonvision_keys"], "seed": config["seed"],
                           "training_config": config, "original_runtime_sha256": digest(Path(runtime.__file__)),
                           "load_report": report, "precision": "fp32",
                           "caution": "Not a fully pretrained CIR checkpoint; random heads are part of this initialization control"}

    def gallery(self, images, captions):
        image, _ = self.model.extract_target_features(images)
        text = self.model.extract_caption_features([clean_caption(t) for t in captions])
        return {"image": image, "caption": text}

    def scores(self, images, texts, targets):
        reference = self.model.ln_vision(self.model.visual_encoder(images))
        combined, visual = self.model.inference(reference, targets["image"], [clean_caption(t) for t in texts], targets["caption"])
        shape = (len(texts), len(targets["image"]))
        return {self.methods[0]: visual.reshape(shape), self.methods[1]: combined.reshape(shape)}
