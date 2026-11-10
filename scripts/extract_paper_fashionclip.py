"""Extract resumable train/val FashionCLIP features from the audited UniFashion image bundle."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from PIL import Image
from src.fashioniq_evaluation import digest, normalize, protocol
from src.paper_comparison import lock_run, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--eval-data", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run feature extraction on the Colab GPU")
    from transformers import CLIPModel, CLIPProcessor
    from huggingface_hub import HfApi
    spec = json.loads((ROOT / "configs/paper_comparisons.json").read_text())["fashionclip"]
    revision = HfApi().model_info(spec["model"], revision=spec["revision"]).sha
    queries, galleries, fingerprint = protocol(args.eval_data)
    meta = pd.read_csv(args.metadata, dtype=str, keep_default_na=False)
    meta = meta[meta.category.isin(["dress", "shirt", "toptee"]) & meta.split.isin(["train", "val"])].reset_index(drop=True)
    if len(meta) != 24016:
        raise ValueError("Expected 18,000 train + 6,016 validation queries")
    from src.paper_comparison import KEYS
    merged = queries.merge(meta[meta.split == "val"], on=KEYS+["text"], how="inner", validate="one_to_one")
    if len(merged) != len(queries):
        raise ValueError("Feature text differs from evaluation protocol")
    train = pd.read_csv(args.data_dir / "train.csv", dtype=str)
    if set(map(tuple, train[KEYS].values)) != set(map(tuple, meta[meta.split == "train"][KEYS].values)):
        raise ValueError("Training metadata differs from audited training data")
    ids = sorted(set(meta.candidate) | set(meta.target) | set().union(*map(set, galleries.values())))
    manifest = json.loads((args.data_dir / "images_manifest.json").read_text())
    for name in ids:
        if digest(args.images_dir / manifest[name]["file"]) != manifest[name]["sha256"]:
            raise ValueError("Image bytes differ: " + name)
    provenance = {"model": spec["model"], "revision": revision, "precision": "fp32",
                  "batch_size": args.batch_size, "protocol": fingerprint, "pixels": "same audited resized files as UniFashion",
                  "images_manifest_sha256": digest(args.data_dir / "images_manifest.json"),
                  "metadata_sha256": digest(args.metadata), "code_sha256": digest(Path(__file__)),
                  "transform": "model's CLIPProcessor (resize/center crop); same image bytes, model-specific transform",
                  "torch": str(torch.__version__)}
    lock_run(args.output, provenance)
    model = CLIPModel.from_pretrained(spec["model"], revision=revision).float().cuda().eval()
    processor = CLIPProcessor.from_pretrained(spec["model"], revision=revision)
    chunks = args.output / "chunks"
    chunks.mkdir(exist_ok=True)
    for kind, items in (("image", ids), ("text", meta.text.tolist())):
        parts = []
        for start in range(0, len(items), 256):
            path = chunks / f"{kind}_{start:06d}.npy"
            seal = path.with_suffix(".json")
            if path.exists() and seal.exists() and digest(path) == json.loads(seal.read_text())["sha256"]:
                parts.append(np.load(path))
                continue
            features = []
            with torch.inference_mode():
                for offset in range(start, min(start+256, len(items)), args.batch_size):
                    batch = items[offset:min(offset+args.batch_size, start+256)]
                    if kind == "image":
                        pixels = []
                        for name in batch:
                            with Image.open(args.images_dir / manifest[name]["file"]) as image:
                                pixels.append(image.convert("RGB"))
                        inputs = processor(images=pixels, return_tensors="pt").to("cuda")
                        feats = model.get_image_features(**inputs)
                    else:
                        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77).to("cuda")
                        feats = model.get_text_features(**inputs)
                    features.append(normalize(feats.float().cpu().numpy()))
            array = np.concatenate(features)
            with path.with_suffix(".tmp").open("wb") as stream:
                np.save(stream, array)
            path.with_suffix(".tmp").replace(path)
            write_json(seal, {"sha256": digest(path)})
            parts.append(array)
            print(f"{kind}: {min(start+256,len(items))}/{len(items)}", flush=True)
        np.save(args.output / f"fashionclip_{kind}_embeddings.npy", np.concatenate(parts))
    pd.DataFrame({"image_id": ids}).to_csv(args.output / "fashionclip_image_ids.csv", index=False)
    meta.to_csv(args.output / "fashionclip_text_metadata.csv", index=False)
    write_json(args.output / "features.json", provenance)
    print("Feature extraction complete; ready for the three-seed fusion controls.")


if __name__ == "__main__":
    main()
