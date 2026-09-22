
import numpy as np
import pandas as pd

from baseline_retrieval import compose_query_vector, build_target_pool, retrieve_topk, recall_at_k

FASHIONCLIP_IMAGE_IDS_CSV = "./logs/fashionclip_image_ids.csv"
FASHIONCLIP_IMAGE_EMB_NPY = "./logs/fashionclip_image_embeddings.npy"
FASHIONCLIP_TEXT_META_CSV = "./logs/fashionclip_text_metadata.csv"
FASHIONCLIP_TEXT_EMB_NPY = "./logs/fashionclip_text_embeddings.npy"
SBERT_TEXT_META_CSV = "./logs/sbert_text_metadata.csv"
SBERT_TEXT_EMB_NPY = "./logs/sbert_text_embeddings.npy"

RESNET50_IMAGE_EMB_NPZ = "./logs/resnet50_image_embeddings.npz"
CLIP_IMAGE_EMB_NPZ = "./logs/clip_image_embeddings.npz"


def load_image_lookup_npy(ids_csv_path, embeddings_npy_path):
    """Trả về (id_to_row: dict, embeddings: ndarray). Dùng cho file kiểu Hiển
    (ảnh) — 1 file .npy embedding + 1 file .csv liệt kê image_id theo đúng
    thứ tự dòng của .npy."""
    ids_df = pd.read_csv(ids_csv_path)
    embeddings = np.load(embeddings_npy_path)
    assert len(ids_df) == embeddings.shape[0], (
        f"{ids_csv_path} có {len(ids_df)} dòng nhưng {embeddings_npy_path} có "
        f"{embeddings.shape[0]} dòng — không khớp, kiểm tra lại 2 file có đúng cặp không."
    )
    id_to_row = {img_id: i for i, img_id in enumerate(ids_df["image_id"])}
    return id_to_row, embeddings


def load_triplet_text_embeddings(metadata_csv_path, embeddings_npy_path, split):
    """Lọc metadata về đúng split + category thật (bỏ category=='all'), trả về
    (metadata đã lọc, embeddings đã lọc THEO ĐÚNG VỊ TRÍ gốc). Dùng .index
    (không phải vị trí sau khi lọc) để tránh lấy nhầm dòng."""
    meta = pd.read_csv(metadata_csv_path)
    embeddings = np.load(embeddings_npy_path)
    assert len(meta) == embeddings.shape[0], (
        f"{metadata_csv_path} có {len(meta)} dòng nhưng {embeddings_npy_path} có "
        f"{embeddings.shape[0]} dòng — không khớp."
    )
    filtered = meta[(meta["category"] != "all") & (meta["split"] == split)]
    filtered_embeddings = embeddings[filtered.index.values]
    return filtered, filtered_embeddings


# ---------------------------------------------------------------------------
# Loader cho định dạng của Diệu (.npz với ids/embeddings/valid)
# ---------------------------------------------------------------------------
def load_image_lookup_npz(npz_path):
    """Trả về (id_to_row: dict, embeddings: ndarray) từ file .npz kiểu Diệu
    (xem extract_image_embeddings_resnet_clip.py — key 'ids'/'embeddings')."""
    data = np.load(npz_path, allow_pickle=True)
    ids = data["ids"]
    id_to_row = {img_id: i for i, img_id in enumerate(ids)}
    return id_to_row, data["embeddings"]


# ---------------------------------------------------------------------------
# Chạy 1 baseline (additive hoặc concat) trên split VAL, per-category pool
# ---------------------------------------------------------------------------
def evaluate_baseline(cand_id_to_row, cand_img_embs_all,
                       target_id_to_row, target_img_embs_all,
                       text_meta, text_embs,
                       method, text_dim=None, k_list=(10, 50)):
    cand_row_idx = text_meta["candidate"].map(cand_id_to_row)
    assert cand_row_idx.isna().sum() == 0, "Có candidate id không tra được embedding ảnh."
    cand_img_embs = cand_img_embs_all[cand_row_idx.values]

    queries = compose_query_vector(cand_img_embs, text_embs, method=method)

    per_category = {}
    for cat in sorted(text_meta["category"].unique()):
        cat_mask = (text_meta["category"] == cat).values
        cat_rows = text_meta[cat_mask]

        target_ids_unique = sorted(cat_rows["target"].unique())
        target_row_idx = [target_id_to_row[t] for t in target_ids_unique]
        target_embs = target_img_embs_all[target_row_idx]
        pool = build_target_pool(target_embs, method=method, text_dim=text_dim)
        pool_ids = np.array(target_ids_unique)

        k_max = max(k_list)
        topk_ids, _ = retrieve_topk(queries[cat_mask], pool, pool_ids, k=k_max)
        true_targets = cat_rows["target"].values

        per_category[cat] = {
            "n_query": len(cat_rows),
            "pool_size": len(pool_ids),
            **{f"recall@{k}": recall_at_k(topk_ids, true_targets, k) for k in k_list},
        }
    return per_category


def _print_report(title, per_category, k_list=(10, 50)):
    print(f"\n=== {title} ===")
    for cat, stats in per_category.items():
        line = f"  {cat:8s} n_query={stats['n_query']:5d} pool={stats['pool_size']:5d}"
        for k in k_list:
            line += f"  Recall@{k}={stats[f'recall@{k}']:.4f}"
        print(line)
    for k in k_list:
        avg = np.mean([s[f"recall@{k}"] for s in per_category.values()])
        print(f"  Trung bình Recall@{k} = {avg:.4f}")


def main():
    # --- additive: FashionCLIP-ảnh + FashionCLIP-văn bản (cặp align duy nhất) ---
    fc_id_to_row, fc_img_embs = load_image_lookup_npy(
        FASHIONCLIP_IMAGE_IDS_CSV, FASHIONCLIP_IMAGE_EMB_NPY
    )
    fc_val_meta, fc_val_text_embs = load_triplet_text_embeddings(
        FASHIONCLIP_TEXT_META_CSV, FASHIONCLIP_TEXT_EMB_NPY, split="val"
    )
    additive_results = evaluate_baseline(
        fc_id_to_row, fc_img_embs, fc_id_to_row, fc_img_embs,
        fc_val_meta, fc_val_text_embs, method="additive",
    )
    _print_report("ADDITIVE (FashionCLIP-ảnh + FashionCLIP-văn bản)", additive_results)

    # --- concat: FashionCLIP-ảnh + SBERT-văn bản (minh hoạ mức sàn) ---
    sbert_val_meta, sbert_val_text_embs = load_triplet_text_embeddings(
        SBERT_TEXT_META_CSV, SBERT_TEXT_EMB_NPY, split="val"
    )
    concat_results = evaluate_baseline(
        fc_id_to_row, fc_img_embs, fc_id_to_row, fc_img_embs,
        sbert_val_meta, sbert_val_text_embs, method="concat",
        text_dim=sbert_val_text_embs.shape[1],
    )
    _print_report("CONCAT (FashionCLIP-ảnh + SBERT-văn bản)", concat_results)

if __name__ == "__main__":
    main()
