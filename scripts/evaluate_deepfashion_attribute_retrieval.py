
import argparse, hashlib, json, math, os, sys, time, traceback, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

ROOT = Path("/content/khoaluantotnghiep")
sys.path.insert(0, str(ROOT))

from src.unifashion_runtime import build_model
from src.unifashion_training import load_trainable_state
from src.unifashion_sanity_data import preprocess, clean_caption
from scripts.train_unifashion_sanity import autocast_context


class ImageDataset(Dataset):
    def __init__(self, names, images_dir, transform):
        self.names = list(names)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        p = self.images_dir / self.names[idx]
        if not p.is_file():
            raise FileNotFoundError(p)
        with Image.open(p) as im:
            return self.transform(im.convert("RGB"))


def find_image_dir(images_dir, names):
    images_dir = Path(images_dir)
    direct = sum((images_dir / x).is_file() for x in names[:min(100, len(names))])
    if direct:
        return images_dir
    raise RuntimeError(
        "Images are not directly inside IMAGES_DIR. "
        "Point IMAGES_DIR to the folder containing the image files."
    )


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv_checked(path, label, required_columns):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(
            f"{label} is 0 bytes: {path}. Input preparation was interrupted; rerun notebook from Cell 7.3."
        )
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise RuntimeError(
            f"{label} has no CSV columns: {path}. Rerun notebook from Cell 7.3."
        ) from error
    if df.empty:
        raise RuntimeError(f"{label} has zero rows: {path}")
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise RuntimeError(f"{label} missing columns: {sorted(missing)}")
    return df


def atomic_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_csv(path, df):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_status(output, **kwargs):
    status_path = Path(output) / "status.json"
    old = {}
    if status_path.is_file():
        try:
            old = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            old = {}
    old.update(kwargs)
    old["updated_at_unix"] = time.time()
    atomic_json(status_path, old)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--captions", type=Path, required=True)
    p.add_argument("--gallery-csv", type=Path, required=True)
    p.add_argument("--queries-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--gallery-chunk-size", type=int, default=256)
    p.add_argument("--query-chunk-size", type=int, default=20)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if args.batch_size < 1 or args.gallery_chunk_size < 1 or args.query_chunk_size < 1:
        raise ValueError("batch/chunk sizes must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    gallery_cache = args.output / "gallery_cache"
    query_cache = args.output / "query_cache"
    gallery_cache.mkdir(exist_ok=True)
    query_cache.mkdir(exist_ok=True)

    # Đọc và kiểm tra input TRƯỚC fingerprint. Không bao giờ fingerprint một CSV rỗng/hỏng.
    gallery_required = {
        "image", "gender", "category", "sleeve", "neckline",
        "upper_fabric", "upper_pattern"
    }
    query_required = {
        "tier", "reference", "target_seed", "text", "gender", "category",
        "sleeve", "neckline", "upper_fabric", "upper_pattern", "positive_count"
    }
    gallery = read_csv_checked(args.gallery_csv, "gallery CSV", gallery_required)
    queries = read_csv_checked(args.queries_csv, "queries CSV", query_required)
    captions = json.loads(args.captions.read_text(encoding="utf-8"))

    names = gallery["image"].astype(str).tolist()
    if len(names) != len(set(names)):
        raise RuntimeError("gallery CSV contains duplicate image names")
    if queries["reference"].astype(str).duplicated().all() and len(queries) > 1:
        print("WARNING: many repeated references may be valid, continuing", flush=True)
    if queries["text"].astype(str).str.strip().eq("").any():
        raise RuntimeError("queries CSV contains empty text")

    images_dir = find_image_dir(args.images_dir, names)

    # Fingerprint v3 tách gallery cache và query cache.
    # Đổi query metadata không làm mất gallery embeddings đã trích hàng giờ.
    gallery_semantic = {
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "gallery_csv_sha256": sha256_file(args.gallery_csv),
        "captions_sha256": sha256_file(args.captions),
    }
    query_semantic = {
        **gallery_semantic,
        "queries_csv_sha256": sha256_file(args.queries_csv),
    }
    fingerprint = {
        "schema": 3,
        "gallery_semantic": gallery_semantic,
        "query_semantic": query_semantic,
        "layout": {
            "batch_size": args.batch_size,
            "gallery_chunk_size": args.gallery_chunk_size,
            "query_chunk_size": args.query_chunk_size,
        },
    }
    fp_path = args.output / "resume_fingerprint.json"

    def previous_parts(old_fp):
        if not isinstance(old_fp, dict):
            return {}, {}
        if old_fp.get("schema") == 3:
            return old_fp.get("gallery_semantic", {}), old_fp.get("query_semantic", {})
        sem = old_fp.get("semantic", old_fp)
        if not isinstance(sem, dict):
            return {}, {}
        old_gallery = {
            "checkpoint_sha256": sem.get("checkpoint_sha256"),
            "gallery_csv_sha256": sem.get("gallery_csv_sha256"),
            "captions_sha256": sem.get("captions_sha256"),
        }
        old_query = {**old_gallery, "queries_csv_sha256": sem.get("queries_csv_sha256")}
        return old_gallery, old_query

    if fp_path.is_file():
        try:
            old_fp = json.loads(fp_path.read_text(encoding="utf-8"))
        except Exception:
            old_fp = {}
        old_gallery, old_query = previous_parts(old_fp)

        gallery_same = old_gallery == gallery_semantic
        query_same = old_query == query_semantic

        if gallery_same and query_same:
            print("RESUME FINGERPRINT: gallery/query compatible", flush=True)
        elif gallery_same and not query_same:
            # Gallery embeddings vẫn hợp lệ; chỉ cất query cache cũ nếu có.
            query_progress = list(query_cache.glob("query_*.done.json")) + list(query_cache.glob("query_*.csv"))
            if query_progress:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                backup = args.output / f"query_cache_backup_{stamp}"
                shutil.move(str(query_cache), str(backup))
                query_cache.mkdir(exist_ok=True)
                print(f"RESUME: query input changed; kept gallery cache, old query cache -> {backup}", flush=True)
            else:
                print("RESUME: query input changed; gallery cache retained", flush=True)
        else:
            # Gallery semantic input thay đổi: cache embeddings cũ không còn an toàn.
            progress = list(gallery_cache.glob("gallery_*.pt")) + list(query_cache.glob("query_*"))
            if progress:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                backup = args.output / f"incompatible_cache_backup_{stamp}"
                backup.mkdir(parents=True, exist_ok=False)
                for d in (gallery_cache, query_cache):
                    if d.exists():
                        shutil.move(str(d), str(backup / d.name))
                gallery_cache.mkdir(exist_ok=True)
                query_cache.mkdir(exist_ok=True)
                print(f"RESUME: gallery semantic changed; old caches preserved at {backup}", flush=True)
            else:
                print("RESUME: stale fingerprint without completed cache -> refreshed", flush=True)
        atomic_json(fp_path, fingerprint)
    else:
        atomic_json(fp_path, fingerprint)
        print("RESUME FINGERPRINT: created", flush=True)

    missing_caps = [x for x in names if x not in captions]
    if missing_caps:
        raise RuntimeError(f"Gallery has {len(missing_caps)} images without captions")

    total_gallery = len(names)
    total_queries = len(queries)
    print(f"TOTAL GALLERY = {total_gallery}", flush=True)
    print(f"TOTAL QUERIES = {total_queries}", flush=True)
    print(f"QUERY CSV BYTES = {args.queries_csv.stat().st_size}", flush=True)

    print("PHASE checkpoint_load", flush=True)
    try:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    except Exception as error:
        print(f"Checkpoint mmap load failed ({type(error).__name__}: {error}); retry without mmap", flush=True)
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = dict(saved["config"])
    checkpoint_meta = {
        "epoch": saved.get("epoch"),
        "monitor_loss": saved.get("monitor_loss"),
        "selection": saved.get("selection"),
    }
    fusion_keys = [k for k in saved["model"] if k.startswith("attention_fusion.")]
    if not fusion_keys:
        raise RuntimeError("No attention_fusion weights in checkpoint")

    device = torch.device("cuda")
    print("PHASE model_build", flush=True)
    model, build_report = build_model(args.upstream, args.assets, config, device)
    load_trainable_state(model, saved["model"])
    del saved
    model.eval()
    print("MODEL READY", flush=True)

    transform = preprocess(config)
    t0 = time.monotonic()

    # ------------------------------------------------------------------
    # 1) GALLERY: cache theo chunk. Mỗi chunk hoàn tất được lưu ngay lên Drive.
    # ------------------------------------------------------------------
    gallery_image_cpu = None
    gallery_caption_cpu = None
    n_gallery_chunks = math.ceil(total_gallery / args.gallery_chunk_size)
    write_status(
        args.output,
        status="gallery_running",
        retrieval_evaluated=False,
        gallery_size=total_gallery,
        queries=total_queries,
        gallery_completed=0,
        query_completed=0,
    )

    with torch.inference_mode():
        for chunk_idx, start in enumerate(range(0, total_gallery, args.gallery_chunk_size), 1):
            end = min(start + args.gallery_chunk_size, total_gallery)
            chunk_names = names[start:end]
            chunk_path = gallery_cache / f"gallery_{start:06d}_{end-1:06d}.pt"

            if chunk_path.is_file():
                chunk = torch.load(chunk_path, map_location="cpu", weights_only=True)
                if chunk.get("names") != chunk_names:
                    raise RuntimeError(f"Gallery cache mismatch: {chunk_path}")
                print(
                    f"GALLERY {end}/{total_gallery} | chunk {chunk_idx}/{n_gallery_chunks} | RESUME CACHE",
                    flush=True,
                )
            else:
                loader = DataLoader(
                    ImageDataset(chunk_names, images_dir, transform),
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                )
                image_feats, caption_feats = [], []
                local_seen = 0

                for batch in loader:
                    ids = chunk_names[local_seen:local_seen + len(batch)]
                    with autocast_context(device, config["precision"]):
                        tf, _ = model.extract_target_features(batch.to(device))
                        cf = model.extract_caption_features(
                            [clean_caption(captions[x]) for x in ids]
                        )
                    image_feats.append(tf.float().cpu())
                    caption_feats.append(cf.float().cpu())
                    local_seen += len(batch)

                    absolute_seen = start + local_seen
                    if absolute_seen == 1 or absolute_seen % 50 == 0 or absolute_seen == total_gallery:
                        pct = 100.0 * absolute_seen / max(1, total_gallery)
                        print(
                            f"GALLERY {absolute_seen}/{total_gallery} | {pct:6.2f}% | "
                            f"chunk {chunk_idx}/{n_gallery_chunks}",
                            flush=True,
                        )

                chunk = {
                    "names": chunk_names,
                    "image_features": torch.cat(image_feats),
                    "caption_features": torch.cat(caption_feats),
                }
                atomic_torch_save(chunk_path, chunk)
                print(
                    f"GALLERY CHECKPOINT SAVED {end}/{total_gallery} -> {chunk_path.name}",
                    flush=True,
                )

            if len(chunk["names"]) != end - start:
                raise RuntimeError(f"Gallery cache row-count mismatch: {chunk_path}")
            if len(chunk["image_features"]) != end - start or len(chunk["caption_features"]) != end - start:
                raise RuntimeError(f"Gallery cache feature-count mismatch: {chunk_path}")

            if gallery_image_cpu is None:
                gallery_image_cpu = torch.empty(
                    (total_gallery,) + tuple(chunk["image_features"].shape[1:]),
                    dtype=chunk["image_features"].dtype,
                )
                gallery_caption_cpu = torch.empty(
                    (total_gallery,) + tuple(chunk["caption_features"].shape[1:]),
                    dtype=chunk["caption_features"].dtype,
                )
            gallery_image_cpu[start:end].copy_(chunk["image_features"])
            gallery_caption_cpu[start:end].copy_(chunk["caption_features"])
            del chunk

            write_status(
                args.output,
                status="gallery_running",
                gallery_completed=end,
                gallery_chunks_completed=chunk_idx,
                gallery_chunks_total=n_gallery_chunks,
                query_completed=0,
            )

    if gallery_image_cpu is None or gallery_caption_cpu is None:
        raise RuntimeError("Gallery cache is empty")
    if len(gallery_image_cpu) != total_gallery or len(gallery_caption_cpu) != total_gallery:
        raise RuntimeError("Gallery cache merge size mismatch")

    print("PHASE gallery_to_gpu", flush=True)
    target_features = gallery_image_cpu.to(device)
    target_captions = gallery_caption_cpu.to(device)
    del gallery_image_cpu, gallery_caption_cpu
    torch.cuda.empty_cache()

    write_status(
        args.output,
        status="query_running",
        gallery_completed=total_gallery,
        query_completed=0,
    )
    print("GALLERY COMPLETE. Bắt đầu query scoring.", flush=True)

    # ------------------------------------------------------------------
    # 2) QUERY: lưu kết quả theo chunk. Chunk đã xong sẽ được bỏ qua khi resume.
    # ------------------------------------------------------------------
    name_to_idx = {n: i for i, n in enumerate(names)}

    # Dựng ground-truth lookup một lần, tránh quét 40k+ gallery cho từng query.
    tier_cols = {
        "fabric_pattern": ["upper_fabric", "upper_pattern"],
        "sleeve_fabric_pattern": ["sleeve", "upper_fabric", "upper_pattern"],
        "detailed": ["sleeve", "neckline", "upper_fabric", "upper_pattern"],
    }
    positive_lookup = {}
    for tier_name, attrs in tier_cols.items():
        grouped = gallery.groupby(["gender", "category"] + attrs, sort=False).groups
        tier_map = {}
        for raw_key, indices in grouped.items():
            if not isinstance(raw_key, tuple):
                raw_key = (raw_key,)
            key = (str(raw_key[0]), str(raw_key[1])) + tuple(int(x) for x in raw_key[2:])
            tier_map[key] = tuple(int(i) for i in indices)
        positive_lookup[tier_name] = tier_map
    print("GROUND TRUTH LOOKUP READY", flush=True)

    n_query_chunks = math.ceil(total_queries / args.query_chunk_size)

    with torch.inference_mode():
        for chunk_idx, start in enumerate(range(0, total_queries, args.query_chunk_size), 1):
            end = min(start + args.query_chunk_size, total_queries)
            chunk_rows = queries.iloc[start:end].reset_index(drop=True)
            result_path = query_cache / f"query_{start:06d}_{end-1:06d}.csv"
            done_path = query_cache / f"query_{start:06d}_{end-1:06d}.done.json"

            if done_path.is_file() and result_path.is_file():
                try:
                    done = json.loads(done_path.read_text(encoding="utf-8"))
                    cached_rows = len(pd.read_csv(result_path))
                    valid_done = (
                        done.get("start") == start
                        and done.get("end") == end
                        and cached_rows == 2 * (end - start)
                    )
                except Exception:
                    valid_done = False
                if valid_done:
                    print(
                        f"QUERY {end}/{total_queries} | chunk {chunk_idx}/{n_query_chunks} | RESUME CACHE",
                        flush=True,
                    )
                    write_status(
                        args.output,
                        status="query_running",
                        gallery_completed=total_gallery,
                        query_completed=end,
                        query_chunks_completed=chunk_idx,
                        query_chunks_total=n_query_chunks,
                    )
                    continue
                print(f"QUERY CACHE INVALID -> recompute {start}:{end}", flush=True)

            qloader = DataLoader(
                ImageDataset(chunk_rows["reference"].tolist(), images_dir, transform),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
            )
            chunk_records = []
            local_seen = 0

            for batch in qloader:
                rows = chunk_rows.iloc[local_seen:local_seen + len(batch)]
                texts = [clean_caption(x) for x in rows["text"].tolist()]
                ref = model.ln_vision(model.visual_encoder(batch.to(device)))

                with autocast_context(device, config["precision"]):
                    combined, visual = model.inference(
                        ref, target_features, texts, target_captions
                    )

                visual = visual.reshape(len(rows), total_gallery).float().cpu().numpy()
                combined = combined.reshape(len(rows), total_gallery).float().cpu().numpy()

                for local_i, (_, row) in enumerate(rows.iterrows()):
                    global_query_index = start + local_seen + local_i
                    tier = str(row["tier"])
                    if tier not in tier_cols:
                        raise RuntimeError(f"Unknown query tier: {tier}")
                    key = (str(row["gender"]), str(row["category"])) + tuple(
                        int(row[attr]) for attr in tier_cols[tier]
                    )
                    pos_idx = set(positive_lookup[tier].get(key, ()))
                    ref_idx = name_to_idx.get(str(row["reference"]))
                    if ref_idx is not None:
                        pos_idx.discard(ref_idx)
                    if not pos_idx:
                        raise RuntimeError(
                            f"Query {global_query_index+1} has no positive inside gallery"
                        )

                    declared_count = int(row["positive_count"])
                    if declared_count != len(pos_idx):
                        # Reference được chọn khác thuộc tính query nên bình thường không nằm trong positives.
                        # Cho phép chênh 1 chỉ để phòng trường hợp protocol tương lai cho reference trùng signature.
                        if abs(declared_count - len(pos_idx)) > 1:
                            raise RuntimeError(
                                f"Query {global_query_index+1} positive_count mismatch: "
                                f"declared={declared_count}, reconstructed={len(pos_idx)}"
                            )

                    for score_name, matrix in [
                        ("image_score", visual),
                        ("image_caption_score", combined),
                    ]:
                        scores = matrix[local_i].copy()
                        if ref_idx is not None:
                            scores[ref_idx] = -np.inf
                        order = np.argsort(-scores, kind="stable")
                        ranks = [int(np.flatnonzero(order == j)[0]) + 1 for j in pos_idx]
                        best_rank = min(ranks)
                        rec = {
                            "query_index": global_query_index + 1,
                            "tier": row["tier"],
                            "reference": row["reference"],
                            "target_seed": row["target_seed"],
                            "text": row["text"],
                            "gender": row["gender"],
                            "category": row["category"],
                            "positive_count": len(pos_idx),
                            "score_type": score_name,
                            "best_positive_rank": best_rank,
                            "top50": json.dumps([names[j] for j in order[:50]]),
                        }
                        for k in (1, 5, 10, 50):
                            rec[f"hit@{k}"] = int(best_rank <= k)
                        chunk_records.append(rec)

                    # Tiến độ chính xác theo query, kiểu 1/3000, 2/3000, ...
                    done_queries = global_query_index + 1
                    pct = 100.0 * done_queries / max(1, total_queries)
                    print(
                        f"QUERY {done_queries}/{total_queries} | {pct:6.2f}% | "
                        f"tier={row['tier']} | ref={row['reference']}",
                        flush=True,
                    )

                local_seen += len(batch)

            chunk_df = pd.DataFrame(chunk_records)
            atomic_csv(result_path, chunk_df)
            atomic_json(done_path, {
                "start": start,
                "end": end,
                "query_count": end - start,
                "record_count": len(chunk_df),
            })
            write_status(
                args.output,
                status="query_running",
                gallery_completed=total_gallery,
                query_completed=end,
                query_chunks_completed=chunk_idx,
                query_chunks_total=n_query_chunks,
            )
            print(
                f"QUERY CHECKPOINT SAVED {end}/{total_queries} -> {result_path.name}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # 3) MERGE + METRICS: chỉ chạy khi mọi query chunk đã hoàn tất.
    # ------------------------------------------------------------------
    result_files = []
    for start in range(0, total_queries, args.query_chunk_size):
        end = min(start + args.query_chunk_size, total_queries)
        result_path = query_cache / f"query_{start:06d}_{end-1:06d}.csv"
        done_path = query_cache / f"query_{start:06d}_{end-1:06d}.done.json"
        if not done_path.is_file() or not result_path.is_file():
            raise RuntimeError(f"Missing completed query chunk: {start}:{end}")
        result_files.append(result_path)

    pred = pd.concat([pd.read_csv(x) for x in result_files], ignore_index=True)
    expected_records = total_queries * 2
    if len(pred) != expected_records:
        raise RuntimeError(
            f"Expected {expected_records} prediction records, got {len(pred)}"
        )
    atomic_csv(args.output / "retrieval_per_query.csv", pred)

    summary = (
        pred.groupby(["score_type", "tier"])
        .agg(
            queries=("reference", "size"),
            mean_positive_count=("positive_count", "mean"),
            mean_best_rank=("best_positive_rank", "mean"),
            **{f"Recall@{k}": (f"hit@{k}", "mean") for k in (1, 5, 10, 50)}
        )
        .reset_index()
    )
    for k in (1, 5, 10, 50):
        summary[f"Recall@{k}"] *= 100.0
    atomic_csv(args.output / "metrics.csv", summary)

    provenance = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint_meta["epoch"],
        "checkpoint_monitor_loss": checkpoint_meta["monitor_loss"],
        "checkpoint_selection": checkpoint_meta["selection"],
        "attention_fusion_keys": fusion_keys,
        "gallery_size": total_gallery,
        "queries": total_queries,
        "config": config,
        "seconds_current_session": time.monotonic() - t0,
        "resume_enabled": True,
        "gallery_chunk_size": args.gallery_chunk_size,
        "query_chunk_size": args.query_chunk_size,
        "fingerprint": fingerprint,
        "note": "custom DeepFashion-MultiModal attribute-retrieval protocol; not an official benchmark",
    }
    atomic_json(args.output / "provenance.json", provenance)
    write_status(
        args.output,
        status="evaluation_completed",
        retrieval_evaluated=True,
        gallery_size=total_gallery,
        queries=total_queries,
        gallery_completed=total_gallery,
        query_completed=total_queries,
        gallery_chunks_completed=n_gallery_chunks,
        gallery_chunks_total=n_gallery_chunks,
        query_chunks_completed=n_query_chunks,
        query_chunks_total=n_query_chunks,
    )

    print("EVALUATION COMPLETED", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        tb = traceback.format_exc()
        try:
            if "--output" in sys.argv:
                output = Path(sys.argv[sys.argv.index("--output") + 1])
                output.mkdir(parents=True, exist_ok=True)
                (output / "last_error.txt").write_text(tb, encoding="utf-8")
                write_status(
                    output,
                    status="failed_resumable",
                    retrieval_evaluated=False,
                    error_type=type(error).__name__,
                    error=str(error),
                )
        except Exception:
            pass
        print(tb, file=sys.stderr, flush=True)
        raise
