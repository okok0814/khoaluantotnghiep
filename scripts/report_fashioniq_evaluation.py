"""Compare measured Recall@K under an identical protocol; keep pending results blank."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd


def generate(baselines, evaluation, output):
    results = [json.loads(p.read_text()) for p in sorted(Path(baselines).glob("fashionclip_*.json"))]
    if not results:
        raise ValueError("No measured baseline results")
    if evaluation:
        results += [json.loads(p.read_text()) for p in sorted(Path(evaluation).glob("unifashion_*score.json"))]
    first = results[0]["protocol"]
    if any(r["protocol"] != first for r in results):
        raise ValueError("Cannot compare scores with different queries/galleries/protocols")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([{ "method": r["method"], **row} for r in results for row in r["summary"]])
    baseline = table[(table.method == "fashionclip_additive")].set_index("category")
    for k in (10, 50):
        table[f"delta_R@{k}_pp_vs_additive"] = [row[f"Recall@{k}"] - baseline.loc[row.category, f"Recall@{k}"]
                                              for _, row in table.iterrows()]
    table.to_csv(output / "comparison.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    macro = table[table.category == "MACRO"].set_index("method")
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    macro[["Recall@10", "Recall@50"]].plot.barh(ax=ax)
    ax.set(xlabel="Macro Recall (%)", ylabel="", xlim=(0, 100),
           title="Measured FashionIQ validation recall")
    ax.grid(axis="x", alpha=.2)
    fig.savefig(output / "recall_comparison.png", dpi=160)
    plt.close(fig)
    lines = ["# Đánh giá Recall@10/@50 trên FashionIQ validation", "",
        "Giao thức: toàn bộ 6.016 query; gallery dress 3.817, shirt 6.346, toptee 5.373; "
        "loại ảnh reference khỏi ranking. Recall trình bày theo phần trăm. Macro là trung bình đều ba loại; "
        "micro là trung bình theo query. Khi score bằng nhau, dùng thứ tự gallery chính thức để phá hòa.", "",
        "Raw query text dùng đúng cột text trong metadata FashionCLIP của Tuần 6 cho mọi mô hình. "
        "Mỗi encoder áp dụng normalization/tokenization riêng. UniFashion dùng clean_caption giống pipeline train; "
        "format raw text này khác cách nối bằng 'and' của CSV train và script validation upstream, "
        "được chọn để thống nhất đầu vào với cached baseline.", "",
        "## Kết quả đo được", "",
        "| Method | Category | Queries | R@10 (%) | R@50 (%) | Δ R@10 (pp) | Δ R@50 (pp) |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in table.to_dict("records"):
        lines.append(f"| {row['method']} | {row['category']} | {row['queries']} | "
                     f"{row['Recall@10']:.2f} | {row['Recall@50']:.2f} | "
                     f"{row['delta_R@10_pp_vs_additive']:+.2f} | {row['delta_R@50_pp_vs_additive']:+.2f} |")
    lines += ["", "![Kết quả thực đo](recall_comparison.png)", "",
        "Δ tính theo điểm phần trăm so với FashionCLIP additive cùng category, không phải phần trăm tăng tương đối.", "",
        "## Nguồn baseline và giới hạn", "",
        "Baseline additive theo notebook `38_18_FashionIQ_RecallK.ipynb`: normalize(image + text), cosine với gallery. "
        "Notebook này đã dùng gallery chính thức và loại reference. Script cũ `evaluate_baseline_fashioniq.py` "
        "dùng gallery target-only, nên không lấy số từ script đó để trộn vào bảng. Các kết quả ở đây được tính lại "
        "từ embedding hiện có, có SHA-256 nguồn và CSV rank từng query.", "",
        "Image-only và text-only là ablation bổ sung. Concat ảnh + SBERT với target zero-padding trong script cũ "
        "không làm text thay đổi thứ hạng cosine: tương đương image-only về ranking, không coi là fusion học được.", "",
        "Cached FashionCLIP embeddings dùng preprocessing theo mô hình của lần trích xuất cũ; chưa đối chiếu pixel "
        "ảnh gốc từ máy thành viên với bộ ảnh resized hiện tại. Đây là so sánh các hệ thống trên cùng "
        "query/gallery, chưa phải kiểm soát hoàn toàn ảnh nguồn hoặc tái lập số liệu paper."]
    fine = [r for r in results if r["method"].startswith("unifashion_")]
    lines += ["", "## Checkpoint UniFashion và diễn giải", ""]
    if not fine:
        lines += ["**Chưa có kết quả Recall UniFashion trong thư mục đầu vào.** Không thể kết luận mô hình "
                  "fine-tune tốt hơn baseline. Người chạy thông báo FP32 tới epoch 4, step 29.500; "
                  "mốc checkpoint dùng đánh giá sẽ đọc trực tiếp từ snapshot, không suy ra từ thông báo này."]
    else:
        for r in fine:
            p = r["provenance"]
            macro = next(row for row in r["summary"] if row["category"] == "MACRO")
            if p.get("tuning"):
                tuning = p["tuning"]
                lines += [f"- {r['method']}: trial `{tuning['trial']['name']}`, "
                          f"parent SHA-256 `{tuning['parent_sha256']}`; "
                          f"{tuning['trial_updates']} update trên {tuning['trial_examples']} mẫu, "
                          f"optimizer khởi tạo lại; không tính thành epoch full-data. "
                          f"Macro R@10={macro['Recall@10']:.2f}%, R@50={macro['Recall@50']:.2f}%."]
                continue
            lines += [f"- {r['method']}: checkpoint `{p['snapshot']}`, SHA-256 `{p['checkpoint_sha256']}`; "
                      f"{p['optimizer_steps']} update, {p['completed_epochs']} epoch hoàn tất và "
                      f"{p['completed_batches_in_current_epoch']} batch trong epoch kế tiếp. "
                      f"Macro R@10={macro['Recall@10']:.2f}%, R@50={macro['Recall@50']:.2f}%."]
        lines += ["", "`unifashion_image_score` vẫn dùng query ảnh + mô tả; chỉ bỏ phần score caption gallery. "
                  "`unifashion_image_caption_score` cộng score caption ảnh gallery được phát hành cùng UniFashion, "
                  "áp dụng cho tất cả ứng viên; không đưa caption của ground-truth target vào query và không ITM rerank.", "",
                  "Đây là đánh giá checkpoint ở tiến độ ghi trên snapshot; không gọi là hoàn tất 10 epoch nếu chưa đủ. "
                  "Nếu chọn checkpoint dựa trên val, cần công bố quy trình lựa chọn; không diễn giải val như test độc lập. "
                  "Chưa có so sánh UniFashion trước/sau fine-tune nên chênh lệch với FashionCLIP chưa cô lập tác dụng fine-tune."]
    (output / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(table[table.category == "MACRO"].to_string(index=False))
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baselines", type=Path, required=True)
    p.add_argument("--evaluation", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    generate(args.baselines, args.evaluation, args.output)


if __name__ == "__main__":
    main()
