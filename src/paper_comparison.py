"""Common evidence utilities for the conference comparisons (no model imports)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.fashioniq_evaluation import CATEGORIES, digest, normalize, protocol, rank_batch, save_result

KEYS = ["category", "candidate", "target"]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def lock_run(output, specification):
    """Resume only an identical run; changed inputs must use a different directory."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "run_spec.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != specification:
            raise ValueError(f"Run inputs/code changed: use a new output directory, preserving {output}")
    else:
        if any(output.iterdir()):
            raise ValueError(f"Nonempty output has no run specification: {output}")
        write_json(path, specification)


def cached_embeddings(folder, queries=None):
    folder = Path(folder)
    names = ["fashionclip_image_ids.csv", "fashionclip_image_embeddings.npy",
             "fashionclip_text_metadata.csv", "fashionclip_text_embeddings.npy"]
    sources = {name: digest(folder / name) for name in names}
    ids = pd.read_csv(folder / names[0], dtype=str, keep_default_na=False).image_id.tolist()
    images = np.load(folder / names[1], mmap_mode="r")
    metadata = pd.read_csv(folder / names[2], dtype=str, keep_default_na=False)
    texts = np.load(folder / names[3], mmap_mode="r")
    if len(ids) != len(set(ids)) or len(ids) != len(images) or len(metadata) != len(texts):
        raise ValueError("Invalid cached embedding mapping")
    if images.ndim != 2 or texts.ndim != 2 or images.shape[1] != texts.shape[1]:
        raise ValueError("Incompatible embedding dimensions")
    metadata["embedding_row"] = np.arange(len(metadata))
    metadata = metadata[metadata.category.isin(CATEGORIES) & metadata.split.isin(["train", "val"])].copy()
    if metadata.duplicated(KEYS + ["split"]).any():
        raise ValueError("Ambiguous category/split embedding rows")
    lookup = {name: i for i, name in enumerate(ids)}
    if queries is not None:
        rows = queries.merge(metadata[metadata.split == "val"][KEYS + ["text", "embedding_row"]],
                             on=KEYS + ["text"], how="left", validate="one_to_one")
        if rows.embedding_row.isna().any():
            raise ValueError("Cached text differs from common validation queries")
    else:
        rows = None
    feature_provenance = folder / "features.json"
    provenance = {"source_sha256": sources, "features": json.loads(feature_provenance.read_text())
                  if feature_provenance.exists() else {"pixels": "legacy cache; original pixels unverified"}}
    return images, texts, metadata, lookup, rows, provenance


def evaluate_vectors(method, queries, galleries, fingerprint, image_vectors, image_lookup,
                     query_vectors, output, provenance, batch_size=128):
    records = []
    if len(query_vectors) != len(queries):
        raise ValueError("Query feature count differs")
    for category in CATEGORIES:
        indices = np.flatnonzero(queries.category.to_numpy() == category)
        rows = queries.iloc[indices]
        gallery = galleries[category]
        vectors = normalize(image_vectors[[image_lookup[x] for x in gallery]])
        for offset in range(0, len(rows), batch_size):
            part = indices[offset:offset + batch_size]
            scores = normalize(query_vectors[part]) @ vectors.T
            records.extend(rank_batch(scores, queries.iloc[part], gallery))
    return save_result(output, method, records, galleries, fingerprint, provenance)


def paired_bootstrap(a, b=None, repeats=5000, seed=42):
    """Category-stratified paired QUERY bootstrap. Not a training-seed confidence interval.

    Resample the same query indices for both systems and both Recall cutoffs.
    Fixed gallery/checkpoint; ignores dependence between queries sharing an image.
    """
    if repeats < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    a = a.set_index(KEYS).sort_index()
    if b is not None:
        b = b.set_index(KEYS).sort_index()
        if not a.index.equals(b.index):
            raise ValueError("Paired comparison requires identical query identities")
    rng = np.random.default_rng(seed)
    samples = np.zeros((repeats, 2), dtype=np.float64)
    point = np.zeros(2)
    for cat in CATEGORIES:
        values = a.xs(cat)[["hit@10", "hit@50"]].to_numpy(dtype=float)
        if b is not None:
            values -= b.xs(cat)[["hit@10", "hit@50"]].to_numpy(dtype=float)
        point += values.mean(axis=0) * 100 / len(CATEGORIES)
        for start in range(0, repeats, 100):
            end = min(start + 100, repeats)
            draws = rng.integers(len(values), size=(end-start, len(values)))
            samples[start:end] += values[draws].mean(axis=1) * 100 / len(CATEGORIES)
    return [{"metric": f"Recall@{k}", "estimate": float(point[i]),
             "lower95": float(np.quantile(samples[:, i], .025)),
             "upper95": float(np.quantile(samples[:, i], .975)),
             "replicates": repeats, "seed": seed, "unit": "paired category-stratified query"}
            for i, k in enumerate((10, 50))]
