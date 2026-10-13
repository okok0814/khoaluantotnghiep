"""Shared, explicit FashionIQ validation protocol for all compared methods."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

CATEGORIES = ("dress", "shirt", "toptee")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norms < 1e-12):
        raise ValueError("Nonfinite or zero embedding; do not silently discard queries/images")
    return x / norms


def protocol(data):
    data = Path(data)
    queries = pd.read_csv(data / "queries.csv", dtype=str, keep_default_na=False)
    galleries = {c: json.loads((data / f"split.{c}.val.json").read_text()) for c in CATEGORIES}
    if queries.duplicated(["category", "candidate", "target"]).any():
        raise ValueError("Duplicate query keys")
    if set(queries.category) != set(CATEGORIES) or queries.text.str.strip().eq("").any():
        raise ValueError("Missing categories or empty query text")
    for c, ids in galleries.items():
        rows = queries[queries.category == c]
        if len(ids) != len(set(ids)) or not len(rows):
            raise ValueError("Duplicate gallery IDs or empty queries")
        if not (set(rows.candidate) | set(rows.target)).issubset(set(ids)):
            raise ValueError("A reference/target is outside the official gallery")
        if (rows.candidate == rows.target).any():
            raise ValueError("Reference cannot equal the labeled target")
    fingerprint = {"queries_sha256": digest(data / "queries.csv"),
        "gallery_sha256": {c: digest(data / f"split.{c}.val.json") for c in CATEGORIES},
        "split": "val", "exclude_reference": True, "ks": [10, 50],
        "tie_break": "descending score, then official gallery order",
        "query_text": "Week6 FashionCLIP metadata text; identical raw text for all methods",
        "aggregation": "per-category, unweighted category macro, query-weighted micro; percentages"}
    return queries, galleries, fingerprint


def rank_batch(scores, rows, gallery):
    """Return exact one-based ranks and top-50; never include the reference."""
    scores = np.array(scores, dtype=np.float32, copy=True)
    if scores.shape != (len(rows), len(gallery)) or not np.isfinite(scores).all():
        raise ValueError("Invalid/nonfinite score matrix")
    lookup = {name: i for i, name in enumerate(gallery)}
    results = []
    for i, row in enumerate(rows.to_dict("records")):
        if row["candidate"] == row["target"]:
            raise ValueError("Reference equals target")
        ref, target = lookup[row["candidate"]], lookup[row["target"]]
        scores[i, ref] = -np.inf
        order = np.argsort(-scores[i], kind="stable")
        order = order[order != ref]
        rank = int(np.flatnonzero(order == target)[0]) + 1
        results.append({**row, "target_rank": rank, "hit@10": int(rank <= 10),
                        "hit@50": int(rank <= 50),
                        "top50": json.dumps([gallery[j] for j in order[:50]])})
    return results


def summarize(records, galleries):
    df = pd.DataFrame(records)
    if not len(df):
        raise ValueError("Cannot summarize empty predictions")
    rows = []
    for c in CATEGORIES:
        part = df[df.category == c]
        if not len(part):
            raise ValueError(f"No predictions for {c}")
        rows.append({"category": c, "queries": len(part), "gallery_size": len(galleries[c]),
                     **{f"Recall@{k}": 100 * part[f"hit@{k}"].mean() for k in (10, 50)}})
    rows.append({"category": "MACRO", "queries": len(df), "gallery_size": None,
                 **{f"Recall@{k}": float(np.mean([r[f"Recall@{k}"] for r in rows])) for k in (10, 50)}})
    rows.append({"category": "MICRO", "queries": len(df), "gallery_size": None,
                 **{f"Recall@{k}": 100 * df[f"hit@{k}"].mean() for k in (10, 50)}})
    return rows


def save_result(output, method, records, galleries, fingerprint, provenance):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = {"method": method, "protocol": fingerprint, "provenance": provenance,
              "summary": summarize(records, galleries)}
    pd.DataFrame(records).to_csv(output / f"{method}_per_query.csv", index=False)
    (output / f"{method}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
