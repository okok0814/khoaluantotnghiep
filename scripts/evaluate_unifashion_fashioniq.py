"""Evaluate a frozen training-checkpoint snapshot on full FashionIQ validation."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from src.fashioniq_evaluation import CATEGORIES, digest, protocol, rank_batch, save_result
from src.unifashion_runtime import build_model
from src.unifashion_sanity_data import clean_caption, preprocess
from src.unifashion_training import load_trainable_state
from scripts.train_unifashion_sanity import autocast_context


class Images(Dataset):
    def __init__(self, ids, images, manifest, transform):
        self.ids, self.images, self.manifest, self.transform = ids, images, manifest, transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        with Image.open(self.images / self.manifest[self.ids[i]]["file"]) as image:
            return self.transform(image.convert("RGB"))


@torch.no_grad()
def score_queries(model, images, target_features, texts, target_captions):
    # Match the official inference path; do not substitute the training prompt tokens.
    reference = model.ln_vision(model.visual_encoder(images))
    combined, visual = model.inference(reference, target_features,
                                        [clean_caption(t) for t in texts], target_captions)
    # Upstream .squeeze() removes the batch dimension when the final batch has one query.
    shape = (len(texts), len(target_features))
    return combined.reshape(shape), visual.reshape(shape)


def evaluate(args):
    if args.checkpoint.name == "last.pt":
        raise ValueError("Copy last.pt to a separate snapshot before evaluating; do not read an actively changing checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("Use the Colab GPU evaluation notebook")
    if args.batch_size < 1:
        raise ValueError("Evaluation batch size must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    print("Reading and hashing the checkpoint snapshot...", flush=True)
    saved = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    config = saved["config"]
    if not saved.get("history") or "cursor" not in saved:
        raise ValueError("Expected a full-training checkpoint with history/cursor; not a benchmark or best-monitor weights file")
    checkpoint_sha = digest(args.checkpoint)
    queries, galleries, fingerprint = protocol(args.eval_data)
    preparation = json.loads((args.eval_data.parent / "preparation.json").read_text())
    for name, expected in preparation["files"].items():
        if digest(args.eval_data / name) != expected:
            raise ValueError(f"Evaluation input changed: {name}")
    if preparation["protocol"] != fingerprint:
        raise ValueError("Protocol fingerprint mismatch")
    if digest(args.data_dir / "data_audit.json") != saved["fingerprint"]["data_audit"]:
        raise ValueError("Use the same audited image/data bundle as training")
    audit = json.loads((args.data_dir / "data_audit.json").read_text())
    if digest(args.data_dir / "images_manifest.json") != audit["files"]["images_manifest.json"]:
        raise ValueError("Image manifest changed")
    manifest = json.loads((args.data_dir / "images_manifest.json").read_text())
    ids = sorted(set().union(*[set(v) for v in galleries.values()]))
    print(f"Verifying {len(ids)} validation images...", flush=True)
    for name in ids:
        if digest(args.images_dir / manifest[name]["file"]) != manifest[name]["sha256"]:
            raise ValueError(f"Validation image hash mismatch: {name}")
    if digest(args.assets / "download_manifest.json") != saved["fingerprint"]["assets_manifest"]:
        raise ValueError("Use the original training assets manifest")
    asset_manifest = json.loads((args.assets / "download_manifest.json").read_text())
    print("Verifying the frozen EVA-G checkpoint...", flush=True)
    vision_records = [r for r in asset_manifest["files"] if r["path"].replace("\\", "/").endswith("/eva_vit_g.pth")]
    if len(vision_records) != 1 or digest(args.assets / "eva_vit_g.pth") != vision_records[0]["sha256"]:
        raise ValueError("Frozen EVA-G checksum differs from the recorded training asset")
    if digest(args.assets / "pretrain/none_lora_0.pt") != "f272d3a3cc91ae809f844d3e258f6a872be405cb203a550c15c4510c7280b541":
        raise ValueError("Domain-pretraining checksum mismatch")
    provenance = {"checkpoint_sha256": checkpoint_sha, "snapshot": args.checkpoint.name,
        "optimizer_steps": len(saved["history"]), "completed_epochs": len(saved["epochs"]),
        "next_epoch_one_based": saved["cursor"][0] + 1, "completed_batches_in_current_epoch": saved["cursor"][1],
        "planned_epochs": config["epochs"], "training_config": config,
        "training_fingerprint": saved["fingerprint"], "eval_batch_size": args.batch_size,
        "precision": config["precision"], "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "scoring": "Official query_tokens target encoder; max query-to-target-token similarity; optional released gallery-caption similarity",
        "caption_rerank": "No ITM reranking, no labeled target captions used as query input",
        "gallery_captions_sha256": digest(args.eval_data / "gallery_captions.json"),
        "code_sha256": {p.name: digest(p) for p in [Path(__file__), ROOT / "src/fashioniq_evaluation.py",
                                                  ROOT / "src/unifashion_runtime.py",
                                                  ROOT / "src/attention_fusion.py"]}}
    device = torch.device("cuda")
    torch.manual_seed(config["seed"])
    torch.set_num_threads(2)
    print("Snapshot progress:", json.dumps({k: provenance[k] for k in
        ("optimizer_steps", "completed_epochs", "next_epoch_one_based", "completed_batches_in_current_epoch")}), flush=True)
    model, load_report = build_model(args.upstream, args.assets, config, device)
    load_trainable_state(model, saved["model"])
    del saved
    model.eval()
    # Upstream extract_target_features otherwise forces FP16 even for an FP32 run.
    model.maybe_autocast = lambda *a, **kw: autocast_context(device, config["precision"])
    (args.output / "checkpoint_load.json").write_text(json.dumps(load_report, indent=2))
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    captions = json.loads((args.eval_data / "gallery_captions.json").read_text())
    transform = preprocess(config)
    predictions = {"unifashion_image_score": [], "unifashion_image_caption_score": []}
    start = time.monotonic()
    with torch.inference_mode():
        for category in CATEGORIES:
            gallery = galleries[category]
            cat_queries = queries[queries.category == category].reset_index(drop=True)
            loader = DataLoader(Images(gallery, args.images_dir, manifest, transform),
                                batch_size=args.batch_size, num_workers=0)
            features, caption_features = [], []
            seen = 0
            for images in loader:
                batch_ids = gallery[seen:seen+len(images)]
                with autocast_context(device, config["precision"]):
                    feats, _ = model.extract_target_features(images.to(device))
                    text_feats = model.extract_caption_features([clean_caption(captions[category][x]) for x in batch_ids])
                features.append(feats.float().cpu())
                caption_features.append(text_feats.float().cpu())
                seen += len(images)
                if seen % (args.batch_size * 25) == 0 or seen == len(gallery):
                    print(f"{category}: gallery {seen}/{len(gallery)}", flush=True)
            target_features = torch.cat(features).to(device)
            target_captions = torch.cat(caption_features).to(device)
            del features, caption_features
            loader = DataLoader(Images(cat_queries.candidate.tolist(), args.images_dir, manifest, transform),
                                batch_size=args.batch_size, num_workers=0)
            seen = 0
            for images in loader:
                rows = cat_queries.iloc[seen:seen+len(images)]
                with autocast_context(device, config["precision"]):
                    combined, visual = score_queries(model, images.to(device), target_features,
                                                     rows.text.tolist(), target_captions)
                for name, scores in (("unifashion_image_score", visual), ("unifashion_image_caption_score", combined)):
                    predictions[name] += rank_batch(scores.float().cpu().numpy(), rows, gallery)
                seen += len(images)
                if seen % (args.batch_size * 25) == 0 or seen == len(cat_queries):
                    print(f"{category}: queries {seen}/{len(cat_queries)}", flush=True)
            # Persist completed-category raw predictions even if a later category is interrupted.
            import pandas as pd
            for name, records in predictions.items():
                pd.DataFrame(records).to_csv(args.output / f"{name}_partial.csv", index=False)
            del target_features, target_captions
    provenance["evaluation_seconds"] = time.monotonic() - start
    for method, records in predictions.items():
        if len(records) != len(queries):
            raise RuntimeError("Incomplete evaluation; no summary allowed")
        result = save_result(args.output, method, records, galleries, fingerprint, provenance)
        print(method, result["summary"], flush=True)
    (args.output / "status.json").write_text(json.dumps({"status": "evaluation_completed",
        "queries": len(queries), "checkpoint_sha256": checkpoint_sha, "retrieval_evaluated": True}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--eval-data", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--upstream", type=Path, default=ROOT / "third_party/UniFashion")
    p.add_argument("--assets", type=Path, default=ROOT / "checkpoints/unifashion")
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
