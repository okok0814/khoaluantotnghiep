import numpy as np

def l2_normalize(x, eps=1e-8):
    x = np.atleast_2d(np.asarray(x, dtype=np.float32))
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, eps, None)


def compose_query_vector(image_emb, text_emb, method="additive", alpha=1.0, normalize=True):

    image_emb = np.atleast_2d(np.asarray(image_emb, dtype=np.float32))
    text_emb = np.atleast_2d(np.asarray(text_emb, dtype=np.float32))
    assert image_emb.shape[0] == text_emb.shape[0], (
        f"Số dòng ảnh ({image_emb.shape[0]}) và văn bản ({text_emb.shape[0]}) phải khớp nhau."
    )

    if normalize:
        image_emb = l2_normalize(image_emb)
        text_emb = l2_normalize(text_emb)

    if method == "additive":
        assert image_emb.shape[1] == text_emb.shape[1], (
            f"additive yêu cầu 2 embedding CÙNG số chiều và CÙNG không gian đã align "
            f"(vd FashionCLIP-ảnh + FashionCLIP-văn bản). Ở đây ảnh có {image_emb.shape[1]} "
            f"chiều, văn bản có {text_emb.shape[1]} chiều — không cộng được. "
            f"Dùng method='concat' cho các cặp backbone khác chiều/không align."
        )
        query = image_emb + alpha * text_emb
    elif method == "concat":
        query = np.concatenate([image_emb, text_emb], axis=1)
    else:
        raise ValueError(f"method không hợp lệ: {method!r} (chỉ nhận 'additive' hoặc 'concat')")

    if normalize:
        query = l2_normalize(query)
    return query



def build_target_pool(target_image_embs, method="additive", text_dim=None, normalize=True):

    target_image_embs = np.atleast_2d(np.asarray(target_image_embs, dtype=np.float32))
    if normalize:
        target_image_embs = l2_normalize(target_image_embs)

    if method == "additive":
        return target_image_embs
    elif method == "concat":
        assert text_dim is not None, "method='concat' cần truyền text_dim để đệm 0 cho đúng chiều."
        zeros = np.zeros((target_image_embs.shape[0], text_dim), dtype=np.float32)
        pool = np.concatenate([target_image_embs, zeros], axis=1)
        return l2_normalize(pool) if normalize else pool
    else:
        raise ValueError(f"method không hợp lệ: {method!r}")



def retrieve_topk(query_vectors, pool_vectors, pool_ids, k=50):
    """Trả về (topk_ids, topk_scores), mỗi cái shape (N_query, k)."""
    pool_ids = np.asarray(pool_ids)
    sims = query_vectors @ pool_vectors.T  # (N_query, M_pool)
    k = min(k, sims.shape[1])
    topk_idx = np.argsort(-sims, axis=1)[:, :k]
    topk_ids = pool_ids[topk_idx]
    topk_scores = np.take_along_axis(sims, topk_idx, axis=1)
    return topk_ids, topk_scores



def recall_at_k(topk_ids, target_ids, k):
    target_ids = np.asarray(target_ids)
    hits = [target_ids[i] in topk_ids[i][:k] for i in range(len(target_ids))]
    return float(np.mean(hits))


def _self_test():
    rng = np.random.default_rng(0)

    D, N, N_DISTRACT = 16, 6, 20
    ref_imgs = rng.normal(size=(N, D)).astype(np.float32)
    texts = rng.normal(size=(N, D)).astype(np.float32)
    true_targets = l2_normalize(ref_imgs + texts)
    distractors = rng.normal(size=(N_DISTRACT, D)).astype(np.float32)

    pool_embs = np.concatenate([true_targets, distractors], axis=0)
    pool_ids = np.array([f"true_{i}" for i in range(N)] + [f"distractor_{i}" for i in range(N_DISTRACT)])
    true_ids = np.array([f"true_{i}" for i in range(N)])

    queries = compose_query_vector(ref_imgs, texts, method="additive")
    pool = build_target_pool(pool_embs, method="additive")
    topk_ids, _ = retrieve_topk(queries, pool, pool_ids, k=len(pool_ids))

    top1_ids = topk_ids[:, 0]
    assert np.array_equal(top1_ids, true_ids), (
        f"additive: hạng 1 phải đúng bằng target thật khi target = image+text theo "
        f"đúng giả thuyết, nhưng nhận được {top1_ids} thay vì {true_ids}"
    )
    r_at_1 = recall_at_k(topk_ids, true_ids, k=1)
    assert r_at_1 == 1.0, r_at_1

    # additive phải báo lỗi rõ ràng khi 2 chiều không khớp (vd ResNet 2048d + SBERT 384d)
    try:
        compose_query_vector(rng.normal(size=(2, 20)), rng.normal(size=(2, 8)), method="additive")
        raise AssertionError("Lẽ ra phải raise AssertionError khi chiều không khớp cho additive")
    except AssertionError as e:
        assert "additive yêu cầu" in str(e)

    D_img, D_txt = 10, 6
    A = rng.normal(size=(1, D_img)).astype(np.float32)          # ảnh tham chiếu
    modifier_text = rng.normal(size=(1, D_txt)).astype(np.float32)
    B_target_dung = rng.normal(size=(1, D_img)).astype(np.float32)      # đúng theo văn bản, khác hẳn A
    C_distractor = A + 0.05 * rng.normal(size=(1, D_img)).astype(np.float32)  # giống hệt A

    q_concat = compose_query_vector(A, modifier_text, method="concat")
    pool_concat = build_target_pool(
        np.concatenate([B_target_dung, C_distractor], axis=0), method="concat", text_dim=D_txt
    )
    topk_ids_c, topk_scores_c = retrieve_topk(
        q_concat, pool_concat, np.array(["B_dung_theo_van_ban", "C_giong_anh_goc"]), k=2
    )

    assert topk_ids_c[0, 0] == "C_giong_anh_goc", (
        "Nếu dòng này FAIL nghĩa là hành vi toán học của concat+zero-pad đã thay đổi "
        "so với thiết kế — cần xem lại build_target_pool()."
    )

    print("[SELF-TEST PASSED] compose_query_vector, build_target_pool, retrieve_topk, "
          "recall_at_k đều đúng — bao gồm cả tính chất 'concat bỏ qua văn bản' đã ghi ở đầu file.")


if __name__ == "__main__":
    _self_test()
