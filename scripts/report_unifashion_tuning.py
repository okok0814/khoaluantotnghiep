"""Report measured study results, keeping screening and full validation separate."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.unifashion_tuning import verify_study, write_json
from src.fashioniq_evaluation import digest


def checked_result(folder, method, expected_protocol, parent_sha, trial=None, expected_examples=None,
                   study_sha=None, scope=None):
    folder = Path(folder)
    path = folder / f"{method}.json"
    if not path.exists():
        return None
    status = json.loads((folder / "status.json").read_text())
    result = json.loads(path.read_text())
    if status["status"] != "evaluation_completed" or result["protocol"] != expected_protocol:
        raise ValueError(f"Incomplete/mismatched evaluation: {folder}")
    p = result["provenance"]
    if scope and p.get("evaluation_scope") != scope:
        raise ValueError("Screening and full validation must remain separate")
    if status["checkpoint_sha256"] != p["checkpoint_sha256"]:
        raise ValueError("Evaluation status/checkpoint mismatch")
    if trial is None:
        if p["checkpoint_sha256"] != parent_sha or p.get("tuning"):
            raise ValueError("Base evaluation is not the study parent")
    else:
        t = p["tuning"]
        if (t["parent_sha256"] != parent_sha or t["trial"] != trial or
                not t["trial_completed"] or t["trial_examples"] != expected_examples):
            raise ValueError("Trials must share parent, settings and completed example budget")
        if study_sha and t["study_sha256"] != study_sha:
            raise ValueError("Trial used different training/screening inputs")
    macro = next(x for x in result["summary"] if x["category"] == "MACRO")
    return {"name": trial["name"] if trial else "base",
            "R@10": macro["Recall@10"], "R@50": macro["Recall@50"],
            "mean_R": (macro["Recall@10"] + macro["Recall@50"]) / 2,
            "checkpoint_sha256": p["checkpoint_sha256"]}


def choose(screen):
    """Retain unchanged parent on ties or when no trial improves screening."""
    base = next(r for r in screen if r["name"] == "base")
    return max(screen, key=lambda r: (r["mean_R"], r["name"] == "base", r["name"]))["name"]


def generate(root):
    root = Path(root)
    output = root / "report"
    output.mkdir(exist_ok=True)
    manifest = verify_study(root / "inputs")
    config = manifest["config"]
    parent = json.loads((root / "parent.json").read_text())
    method = config["selection_method"]
    trial_states, screen, full = [], [], []
    for trial in [None] + config["trials"]:
        name = trial["name"] if trial else "base"
        if trial:
            status_path = root / "trials" / name / "status.json"
            state = json.loads(status_path.read_text()) if status_path.exists() else {"status": "pending"}
            trial_states.append({"name": name, **trial, "status": state["status"],
                                 "error": state.get("error", "")})
        for stage, expected, collection in [("screen", manifest["screen_protocol"], screen),
                                            ("full", manifest["source_protocol"], full)]:
            row = checked_result(root / stage / name, method, expected, parent["checkpoint_sha256"],
                                 trial, manifest["train_examples"], digest(root / "inputs/study.json"),
                                 "screening_only_reduced_gallery" if stage == "screen" else "full_validation")
            if row:
                collection.append(row)
    for row in full:
        screened = next((r for r in screen if r["name"] == row["name"]), None)
        if screened and screened["checkpoint_sha256"] != row["checkpoint_sha256"]:
            raise ValueError("Screening/full results must use the same checkpoint")
    pd.DataFrame(trial_states).to_csv(output / "trial_status.csv", index=False)
    for name, rows in (("screening", screen), ("full_validation", full)):
        pd.DataFrame(rows, columns=["name", "R@10", "R@50", "mean_R", "checkpoint_sha256"]).to_csv(
            output / (name + ".csv"), index=False)
    measured = {r["name"] for r in screen}
    failed = {r["name"] for r in trial_states if r["status"] == "failed"}
    ready = "base" in measured and all(t["name"] in measured or t["name"] in failed for t in config["trials"])
    selected = choose(screen) if ready else None
    selection = {"ready": ready, "selected": selected, "failed_trials": sorted(failed),
        "selection_method": method, "selection_metric": config["selection_metric"],
        "parent_sha256": parent["checkpoint_sha256"],
        "warning": "Screening selection only; full validation against base is required."}
    write_json(output / "selection.json", selection)
    lines = ["# Tuần 10 — Tinh chỉnh learning rate và batch size", "",
        f"Parent SHA-256: `{parent['checkpoint_sha256']}`. Tiến độ thực tế: `{parent['progress']}`.", "",
        "Mỗi trial khởi tạo lại AdamW từ cùng weights; FP32, LR cố định, cùng 1 lượt trên tập train con "
        f"{manifest['train_examples']} mẫu. Không tiếp tục optimizer/OneCycle của run gốc. "
        "Batch lớn hơn có ít update hơn nhưng cùng số ví dụ; số negative trong batch cũng thay đổi.", "",
        "## Trạng thái trial", "", "| Trial | LR | Batch | Trạng thái |", "|---|---:|---:|---|"]
    for t in trial_states:
        lines.append(f"| {t['name']} | {t['learning_rate']} | {t['batch_size']} | {t['status']} |")
    for title, rows in (("Screening — gallery giảm, không so với paper/Tuần 9", screen),
                        ("Full validation — cùng giao thức Tuần 9", full)):
        lines += ["", f"## {title}", "", "| Model | Macro R@10 (%) | Macro R@50 (%) | Mean R (%) |",
                  "|---|---:|---:|---:|"]
        for r in rows:
            lines.append(f"| {r['name']} | {r['R@10']:.2f} | {r['R@50']:.2f} | {r['mean_R']:.2f} |")
        if not rows:
            lines.append("Chưa có kết quả; không điền số giả hoặc chọn hyperparameter tốt nhất.")
    lines += ["", f"Shortlist: **{selected or 'chưa đủ screening'}**. "
              "Tie chọn base; trial lỗi bị loại và ghi rõ, không tự giảm batch rồi giữ nguyên tên trial.", "",
              "## Đối chiếu số liệu công bố", "",
              "UniFashion, Table 4: macro R@10=58,93%, R@50=76,93%. "
              "[Paper EMNLP 2024](https://aclanthology.org/2024.emnlp-main.89.pdf). "
              "Chênh lệch chỉ là tham chiếu số học: khác caption gallery, text preprocessing, "
              "reference exclusion, precision và batch; chưa phải tái lập protocol paper.", ""]
    base = next((r for r in full if r["name"] == "base"), None)
    tuned = next((r for r in full if r["name"] == selected and selected != "base"), None)
    if base and tuned:
        lines += [f"So với parent cùng thời điểm: Δ R@10={tuned['R@10']-base['R@10']:+.2f} pp; "
                  f"Δ R@50={tuned['R@50']-base['R@50']:+.2f} pp. "
                  f"Khoảng cách số học paper − trial: {58.93-tuned['R@10']:+.2f} / "
                  f"{76.93-tuned['R@50']:+.2f} pp.",
                  "Nếu full validation giảm, giữ base; không tuyên bố tuning thành công chỉ vì screening tăng."]
    else:
        lines += ["Chưa có cặp full-val base/trial được chọn để kết luận cải thiện do tuning. "
                  "Nếu shortlist là base thì chưa tìm được cấu hình tốt hơn ở vòng screening này."]
    lines += ["", "Kết quả một seed và ngân sách ngắn chỉ mang tính thăm dò. Validation được dùng chọn cấu hình; "
              "không gọi là test độc lập. Muốn đánh giá ổn định cần thêm seed/ngân sách. "
              "Báo cáo cả scoring có/không caption trong JSON; tiêu chí chọn đã cố định là scoring có caption.", ""]
    (output / "weekly_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(selection, indent=2))
    return selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    generate(parser.parse_args().root)
