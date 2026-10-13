"""Align the Week6 cached FashionCLIP baselines to a shared val protocol and rerun."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.fashioniq_evaluation import CATEGORIES, digest, normalize, protocol, rank_batch, save_result
from scripts.prepare_unifashion_sanity import caption_lookup


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs", type=Path, default=ROOT / "logs")
    p.add_argument("--data", type=Path, default=ROOT / "checkpoints/week8_data")
    p.add_argument("--upstream", type=Path, default=ROOT / "third_party/UniFashion")
    p.add_argument("--output", type=Path, default=ROOT / "reports/week9_fashioniq_evaluation")
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    data = out / "evaluation_data"
    data.mkdir()
    metadata = pd.read_csv(args.logs / "fashionclip_text_metadata.csv", dtype=str)
    meta = metadata[(metadata.split == "val") & metadata.category.isin(CATEGORIES)].copy()
    meta["embedding_row"] = meta.index
    keys = ["category", "candidate", "target"]
    val = pd.read_csv(args.data / "val.csv", dtype=str)
    if meta.duplicated(keys).any() or val.duplicated(keys).any():
        raise ValueError("Query pairs ambiguous")
    rows = val[keys].merge(meta[keys + ["text", "triplet_id", "embedding_row"]],
                           on=keys, how="outer", validate="one_to_one", indicator=True)
    if len(rows) != 6016 or not rows._merge.eq("both").all():
        raise ValueError("Week6 query metadata differs from audited FashionIQ validation")
    rows.drop(columns=["_merge", "embedding_row"]).to_csv(data / "queries.csv", index=False)
    captions = {}
    for c in CATEGORIES:
        shutil.copyfile(args.data / "galleries" / f"split.{c}.val.json", data / f"split.{c}.val.json")
        lookup, conflicts = caption_lookup(args.upstream / "dataset" / f"next_llava.val.{c}.caption.json")
        ids = json.loads((data / f"split.{c}.val.json").read_text())
        if conflicts or set(ids) - set(lookup):
            raise ValueError(f"Released captions missing/ambiguous for {c}")
        captions[c] = {image_id: lookup[image_id] for image_id in ids}
    (data / "gallery_captions.json").write_text(json.dumps(captions), encoding="utf-8")
    queries, galleries, fingerprint = protocol(data)
    image_ids = pd.read_csv(args.logs / "fashionclip_image_ids.csv", dtype=str).image_id.tolist()
    images = np.load(args.logs / "fashionclip_image_embeddings.npy", mmap_mode="r")
    texts = np.load(args.logs / "fashionclip_text_embeddings.npy", mmap_mode="r")
    if len(set(image_ids)) != len(image_ids) or len(image_ids) != len(images) or len(metadata) != len(texts):
        raise ValueError("Invalid embedding mapping")
    image_rows = {name: i for i, name in enumerate(image_ids)}
    sources = ["fashionclip_image_ids.csv", "fashionclip_image_embeddings.npy",
               "fashionclip_text_metadata.csv", "fashionclip_text_embeddings.npy"]
    provenance = {"source": "Existing Week6/Week37 FashionCLIP embedding artifacts",
        "source_sha256": {name: digest(args.logs / name) for name in sources},
        "reference_notebook": "notebooks/38_18_FashionIQ_RecallK.ipynb",
        "image_preprocessing": "Cached Week6 model-specific preprocessing; original image pixels not reverified",
        "note": "Full official galleries, reference excluded. Baseline rerun, not a new model training run."}
    predictions = {name: [] for name in ("fashionclip_additive", "fashionclip_image_only", "fashionclip_text_only")}
    for c in CATEGORIES:
        cat = rows[rows.category == c].drop(columns=["_merge", "embedding_row"]).reset_index(drop=True)
        original = rows[rows.category == c]
        gallery = galleries[c]
        g = normalize(images[[image_rows[x] for x in gallery]])
        im = np.asarray(images[[image_rows[x] for x in cat.candidate]], dtype=np.float32)
        tx = np.asarray(texts[original.embedding_row.astype(int).to_numpy()], dtype=np.float32)
        # Same additive composition as the Week6 notebook; input artifacts are already normalized.
        vectors = {"fashionclip_additive": normalize(im + tx),
                   "fashionclip_image_only": normalize(im), "fashionclip_text_only": normalize(tx)}
        for method, q in vectors.items():
            for start in range(0, len(cat), 128):
                predictions[method] += rank_batch(q[start:start+128] @ g.T, cat.iloc[start:start+128], gallery)
        print(f"Evaluated baselines: {c}, {len(cat)} queries, {len(gallery)} gallery images", flush=True)
    for method, records in predictions.items():
        result = save_result(out / "baselines", method, records, galleries, fingerprint, provenance)
        print(method, result["summary"][-2], flush=True)
    manifest = {"files": {p.name: digest(p) for p in data.iterdir()}, "protocol": fingerprint,
                "status": "baseline_evaluated_unifashion_pending", "query_counts": queries.category.value_counts().to_dict()}
    (out / "preparation.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
