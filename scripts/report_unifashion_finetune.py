"""Render observed training logs; never infer unrun epochs or Recall@K."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def generate(directory):
    directory = Path(directory)
    status = json.loads((directory / "status.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    train_path, epoch_path = directory / "train_metrics.csv", directory / "epoch_metrics.csv"
    train = pd.read_csv(train_path) if train_path.exists() else pd.DataFrame()
    epochs = pd.read_csv(epoch_path) if epoch_path.exists() else pd.DataFrame()
    # An interrupted session can leave step rows newer than the saved weights.
    checkpoint_step = status.get("checkpoint_step")
    if checkpoint_step is not None and len(train):
        train = train[train.step <= checkpoint_step]
        if len(epochs):
            epochs = epochs[epochs.step <= checkpoint_step]
    if len(train):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        ax = axes[0, 0]
        ax.plot(train.step, train.loss, alpha=.2, linewidth=.6, label="Observed batch loss")
        window = min(100, len(train))
        ax.plot(train.step, train.loss.rolling(window, min_periods=1).mean(),
                label=f"Trailing mean ({window} updates)")
        ax.set(xlabel="Optimizer update", ylabel="ITC + ITM + TTC", title="Training loss")
        ax.legend()
        for key in ("loss_itc", "loss_itm", "loss_ttc"):
            axes[0, 1].plot(train.step, train[key].rolling(window, min_periods=1).mean(), label=key)
        axes[0, 1].set(xlabel="Optimizer update", ylabel="Smoothed loss", title="Loss components")
        axes[0, 1].legend()
        axes[1, 0].plot(train.step, train.lr)
        axes[1, 0].set(xlabel="Optimizer update", ylabel="Learning rate", title="Actual OneCycle schedule")
        before_path = directory / "validation_before.json"
        if before_path.exists():
            before = sum(json.loads(before_path.read_text()).values())
            monitor_x, monitor_y = [0], [before]
        else:
            monitor_x, monitor_y = [], []
        if len(epochs):
            axes[1, 1].plot(epochs.epoch, epochs.train_loss, "o-", label="Train epoch mean")
            monitor_x += epochs.epoch.tolist()
            monitor_y += epochs.monitor_loss.tolist()
        axes[1, 1].plot(monitor_x, monitor_y, "o-", label="Fixed monitor loss")
        axes[1, 1].set(xlabel="Completed epoch (0 = pretrained)", ylabel="Total loss",
                       title="Loss monitoring; not Recall@K")
        axes[1, 1].legend()
        for ax in axes.flat:
            ax.grid(alpha=.2)
        fig.savefig(directory / "training_curves.png", dpi=160)
        plt.close(fig)
    lines = ["# Báo cáo fine-tune UniFashion trên FashionIQ", "",
        f"Trạng thái ghi nhận: `{status['status']}`. Đã ghi nhận {len(epochs)}/{config['epochs']} epoch hoàn tất "
        f"và {len(train)} cập nhật trong log được chọn.", "",
        "Bài toán: truy xuất ảnh đích từ ảnh tham chiếu và câu mô tả thay đổi. "
        "Khởi tạo từ domain-pretraining UniFashion; đóng băng EVA-G; cập nhật Q-Former và các head truy xuất.", "",
        f"Cấu hình: batch {config['batch_size']}, precision {config['precision']}, "
        f"AdamW max LR {config['learning_rate']}, weight decay {config['weight_decay']}; "
        f"OneCycle {config['epochs']} epoch, warm-up {config['warmup_epochs']} epoch. "
        "Đây là cấu hình điều chỉnh theo GPU, không phải tái lập batch 128 của repo.", "",
        "Train dùng 18.000 truy vấn. Monitor cố định dùng 96 truy vấn (32 mỗi loại), "
        "không giao ảnh với gallery train. Loss phụ thuộc batch và tập negative, "
        "không thay thế Recall@10/50 trên toàn bộ gallery FashionIQ.", ""]
    if len(epochs):
        lines += ["| Epoch | Train loss | Monitor loss | ITC monitor | ITM monitor | TTC monitor |",
                  "|---:|---:|---:|---:|---:|---:|"]
        for row in epochs.to_dict("records"):
            lines.append(f"| {int(row['epoch'])} | {row['train_loss']:.6f} | {row['monitor_loss']:.6f} | "
                         f"{row['monitor_loss_itc']:.6f} | {row['monitor_loss_itm']:.6f} | {row['monitor_loss_ttc']:.6f} |")
        best = epochs.loc[epochs.monitor_loss.idxmin()]
        lines += ["", f"Monitor loss thấp nhất trong các epoch đã ghi: {best.monitor_loss:.6f} "
                  f"ở epoch {int(best.epoch)}. `best_monitor.pt` được chọn theo tiêu chí này."]
        if len(epochs) >= 2:
            last, previous = epochs.iloc[-1], epochs.iloc[-2]
            if last.train_loss < previous.train_loss and last.monitor_loss > previous.monitor_loss:
                lines += ["Ở hai epoch cuối, train loss giảm nhưng monitor loss tăng; cần theo dõi thêm "
                          "và đánh giá retrieval trước khi kết luận overfitting."]
    else:
        lines += ["Chưa có epoch hoàn tất trong log này; chưa đủ dữ liệu để kết luận xu hướng huấn luyện."]
    if len(train):
        lines += ["", "![Đường cong từ log thực tế](training_curves.png)", "",
                  f"Số lần thử lại do AMP overflow trong các update đã ghi: {int(train.amp_retries.sum())}. "
                  f"Gradient norm lớn nhất trước clipping: {train.grad_norm_before_clip.max():.6f}."]
        if "seconds" in train:
            lines += [f"Tổng thời gian các bước cập nhật đã ghi: {train.seconds.sum()/3600:.3f} giờ "
                      "(không bao gồm đọc ảnh, tải model, validation, lưu checkpoint và phần chạy lại sau ngắt phiên)."]
    environment = directory / "environment.json"
    if environment.exists():
        env = json.loads(environment.read_text())
        lines += ["", f"GPU của phiên được ghi gần nhất: {env.get('gpu', 'không ghi nhận')}; "
                  f"PyTorch: {env.get('torch', 'không ghi nhận')}; CUDA: {env.get('cuda', 'không ghi nhận')}."]
    lines += ["", "Giới hạn: chưa đánh giá Recall@K hoặc so sánh baseline trong báo cáo này. "
              "Loss giảm không đủ chứng minh chất lượng truy xuất tăng. Khi chưa đủ 10 epoch, "
              "chỉ báo cáo số epoch thực sự hoàn tất; tiếp tục từ last.pt với cùng config.", "",
              "Minh chứng: config.json, environment.json, data_audit.json, batch_plan.json, "
              "checkpoint_load.json, train_metrics.csv, epoch_metrics.csv, status.json và run.log."]
    (directory / "training_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory / "training_report.md"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    print(generate(args.run_dir))


if __name__ == "__main__":
    main()
