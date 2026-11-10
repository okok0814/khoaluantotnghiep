# Tuần 43 — Hoàn tất huấn luyện và lưu checkpoint tốt nhất

**Cập nhật 02/11/2026:** run FP32 gốc hoàn tất 10 epoch / 90.000 update; đã đánh giá full validation
và nhận receipt lưu `best_recall_step90000.pt` (hash và CPU reload tại Colab đã kiểm tra).
Macro R@10/R@50 đạt **64,79%/80,64%**, tốt hơn step60000 trong hai ứng viên đã so sánh.
Nhiệm vụ hoàn tất training/lưu best đã xong; attention-fusion còn chờ kết quả.
Phần tiến độ 78.000 và hướng dẫn lưu step60000 bên dưới là lịch sử ngày 27/10;
không cần thực hiện lại. [Báo cáo đã kiểm tra](../logs/week43_44_unifashion_completion/weekly_report.md).

Cập nhật **27/10/2026**. Kết quả tuning đã được xác nhận bằng full validation: **giữ parent step 60.000**, không chọn trial LR 1e-5/batch 4.

| Checkpoint, scoring có caption gallery | Macro R@10 (%) | Macro R@50 (%) | Mean R (%) |
|---|---:|---:|---:|
| Step 29.500 | 57,08 | 75,43 | 66,25 |
| **Parent step 60.000** | **62,57** | **79,18** | **70,88** |
| Trial LR 1e-5/batch 4 từ step 60.000 | 61,88 | 78,88 | 70,38 |

Tiêu chí đã dùng: trung bình macro Recall@10 và Recall@50 trên full validation, scoring có caption gallery. Đây là tốt nhất **trong các checkpoint đã đánh giá**, không khẳng định tốt nhất của tất cả epoch. [Báo cáo đầy đủ](../logs/week42_unifashion_tuning/full_validation_report.md), [metadata lựa chọn](../logs/week42_unifashion_tuning/best_checkpoint_selection.json).

## Bảo toàn checkpoint step 60.000

Snapshot nằm trong study tuning, khác `last.pt` của run gốc vốn tiếp tục được ghi đè. Chạy cell sau trong Colab (CPU runtime cũng được; chỉ sao chép file trên Drive). Cần dung lượng cho thêm một bản checkpoint. SHA-256 là số ghi trong minh chứng đã đánh giá; cell xác minh file thực tế trước và sau khi copy.

```python
from google.colab import drive
from pathlib import Path
import hashlib, json, shutil

drive.mount('/content/drive')
root = Path('/content/drive/MyDrive/Thesis/UniFashion/week8')
source = root / 'tuning/fp32_lr_batch_01/parent_snapshot.pt'
destination = root / 'selected_checkpoints/best_recall_step60000.pt'
expected = 'df8226a0e61e41ebb5bf88f1870b0176118f75d14612ceb8c0e04cf91309a683'

def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

assert source.is_file(), f'Missing original snapshot: {source}'
assert sha256_file(source) == expected, 'Source differs from the evaluated checkpoint'
destination.parent.mkdir(parents=True, exist_ok=True)
if not destination.exists():
    # Exclusive creation avoids overwriting any checkpoint already saved here.
    with source.open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
assert sha256_file(destination) == expected, 'Copy is incomplete/different; do not use it'
selection = {
    'checkpoint': destination.name, 'checkpoint_sha256': expected,
    'source': str(source), 'optimizer_steps': 60000,
    'scoring_method': 'unifashion_image_caption_score',
    'selection_criterion': 'mean of full-validation macro Recall@10 and Recall@50',
    'Recall@10': 62.57296585419965, 'Recall@50': 79.18236192230523,
    'mean_R': 70.87766388825244,
    'scope': 'Best among step29500, step60000 and lr1e5_b4 with full validation',
    'copy_sha256_verified': True
}
destination.with_suffix('.json').write_text(json.dumps(selection, indent=2))
print('Verified best-checkpoint copy:', destination)
```

Nếu cell bị ngắt lúc copy và lần sau báo checksum sai, file đích chưa đủ; giữ snapshot nguồn, chuyển bản copy lỗi sang tên khác trước khi chạy lại. Không tắt kiểm tra hash để báo thành công. Chỉ đánh dấu “đã lưu bản best riêng” sau khi thấy dòng `Verified best-checkpoint copy`.

## Hoàn tất run gốc

1. Dùng notebook [40_16_UniFashion_Finetune_FashionIQ.ipynb](../notebooks/40_16_UniFashion_Finetune_FashionIQ.ipynb), code/data ZIP gốc và config FP32 gốc trên Drive. Resume `runs/full_train_fp32_b2_seed42/last.pt`; không dùng trial checkpoint và không thay runtime bằng attention-fusion mới trên GitHub.
2. Giữ `STOP_AFTER_EPOCH = 10`. Người chạy báo step 78.000; checkpoint sẽ xác nhận mốc đã lưu. Với 9.000 update/epoch, mục tiêu là 90.000 update, còn khoảng 12.000 nếu checkpoint đúng 78.000.
3. Xác nhận `status.json` của run báo `completed`, `completed_epochs=10` và checkpoint tương ứng. Không chỉ dựa vào việc cell kết thúc.
4. Sao chép `last.pt` đã hoàn thành thành snapshot mới, ví dụ `fp32_step90000.pt`, trong thư mục evaluation mới. Giữ parent step 60.000 và bản best hiện tại.
5. Dùng notebook [41_15_UniFashion_FashionIQ_Recall_Evaluation.ipynb](../notebooks/41_15_UniFashion_FashionIQ_Recall_Evaluation.ipynb) và bundle evaluation gốc đang dùng trên Drive; chạy cùng full-val 6.016 query. Không chạy notebook tạo snapshot mới vào folder study tuning cũ.
6. So sánh mean macro R@10/@50 có caption với **70,87766388825244**. Nếu cao hơn thì lưu snapshot mới cùng metadata; nếu bằng hoặc thấp hơn thì giữ step 60.000 theo quy tắc ưu tiên checkpoint đã chọn khi hòa.

`best_monitor.pt` chọn theo loss trên 96 query monitor, không đảm bảo tốt nhất theo retrieval. Có thể đánh giá nó riêng nếu cần, nhưng pipeline evaluation hiện yêu cầu checkpoint full-training có history/cursor; không đổi tên file rồi giả làm `last.pt`. Không suy ra Recall tốt hơn từ epoch muộn hơn hoặc monitor loss thấp hơn.

## Trạng thái báo cáo

- Đã hoàn tất: ba trial, screening, full-val base/candidate, đối chiếu rank và quyết định giữ base.
- Chưa xác minh: bản sao best riêng trên Drive (cần chạy cell ở trên).
- Đang thực hiện: hoàn tất 10 epoch, đánh giá snapshot cuối và quyết định checkpoint bàn giao.

Giữ checkpoint trên Drive; commit code/config/metric/metadata lên GitHub, không commit `.pt`. Các số full-val hiện cao hơn paper về mặt số học nhưng protocol còn khác nhau; không tuyên bố tái lập/vượt paper hoặc tuning thành công dựa vào điều đó.
