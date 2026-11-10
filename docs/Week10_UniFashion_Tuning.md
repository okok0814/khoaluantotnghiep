# Tuần 10 — Tinh chỉnh learning rate và batch size cho UniFashion

Chuẩn bị **20/10/2026**. Hạng mục: “Tinh chỉnh hyperparameter (learning rate, batch size) cho mô hình đã fine-tune để thu hẹp khoảng cách với số liệu công bố, nếu khả thi.”

Output notebook nhận ngày 20/10 xác nhận snapshot FP32 tại step 60.000: hoàn tất 6 epoch và 6.000 batch epoch 7. LR kế tiếp của OneCycle trong snapshot là khoảng 6,6761e-6.

**Cập nhật 27/10: ba trial, screening và full validation của base/ứng viên đã hoàn thành.** [Báo cáo](../logs/week42_unifashion_tuning/weekly_report.md): base step 60.000 đạt macro R@10/R@50 **62,57% / 79,18%**, cao hơn trial LR 1e-5/batch 4 (**61,88% / 78,88%**); giữ base. Notebook [42_14](../notebooks/42_14_UniFashion_Hyperparameter_Tuning.ipynb) vẫn giữ output lần chạy trước bị ngắt, còn kết quả mới nằm trong ZIP minh chứng đã nhập.

Screening từng chọn `lr1e5_b4`: R@10/R@50 **90,10% / 95,31%**, so với base **88,02% / 96,35%**; Mean R tăng **0,52 pp**, nhưng R@50 giảm **1,04 pp**. Các số dùng gallery giảm, không so paper/full-val. Kết quả xác nhận full-val sau đó giảm **0,50 pp Mean R**, nên shortlist không phải quyết định checkpoint cuối.

**Việc tiếp theo:** không cần chạy lại tuning đã hoàn tất. Bảo toàn parent step 60.000 và tiếp tục run huấn luyện gốc theo [hướng dẫn tuần 43](Week43_UniFashion_Checkpoint_Selection.md). Các mục bên dưới mô tả thiết kế và cách chạy thí nghiệm đã thực hiện, không phải yêu cầu chạy lại.

## Đối chiếu paper trước khi đặt mục tiêu

[UniFashion, EMNLP 2024, Table 4](https://aclanthology.org/2024.emnlp-main.89.pdf) báo cáo:

| Kết quả | Macro R@10 (%) | Macro R@50 (%) |
|---|---:|---:|
| UniFashion trong paper | 58,93 | 76,93 |
| Run dự án step 29.500, có score caption gallery | 57,08 | 75,43 |
| Paper − run dự án (pp, dùng số gốc chưa làm tròn) | 1,85 | 1,50 |

Đây là **khoảng cách số học tham khảo**, không phải khoảng cách tái lập có kiểm soát. Paper, phụ lục C, mô tả caption ảnh evaluation được tạo bởi UniFashion đã train; dự án dùng caption gallery được phát hành trong repo. Ngoài ra còn khác text preprocessing, loại reference, precision và batch. Không đổi các yếu tố này trong phép thử LR/batch rồi quy toàn bộ cải thiện cho hyperparameter.

Config trong [cir_ft.sh ở revision đã ghim](https://github.com/xiangyu-mm/UniFashion/blob/f01a46fb7ed08255fcb52a943d9bcefde03c8eaf/cir_ft.sh) dùng learning rate 2e-5, batch 128, 10 epoch, BF16. Run của dự án dùng batch 2, FP32 và OneCycle với **max LR** 2e-5; LR thực tế tại step 60.000 không nhất thiết còn là 2e-5. Notebook ghi LR kế tiếp của run gốc trong `parent.json`. Không dùng batch 32 của phần LLM trong paper làm thông số CIR.

## Thiết kế phù hợp ngân sách Colab

Tất cả trial bắt đầu độc lập từ **cùng snapshot FP32**; giữ seed 42, preprocessing, frozen EVA-G, loss và scoring như pipeline hiện có. Mỗi trial tạo **AdamW mới, LR cố định**, không nạp optimizer/OneCycle của run gốc. Đây là thử nghiệm fine-tune bổ sung ngắn, không phải resume chính xác lịch huấn luyện 10 epoch.

| Trial | Learning rate cố định | Batch thực | Mẫu train | Update |
|---|---:|---:|---:|---:|
| `base` | Không train thêm | — | 0 | 0 |
| `lr1e5_b2` | 1e-5 | 2 | 1.200 | 600 |
| `lr5e6_b2` | 5e-6 | 2 | 1.200 | 600 |
| `lr1e5_b4` | 1e-5 | 4 | 1.200 | 300 |

- 400 mẫu/category, target không trùng; cùng ví dụ và cùng thứ tự cho mọi trial. Đây là subset thăm dò, không đại diện cho full-data training.
- Ngân sách bằng nhau theo **số ví dụ đã xử lý**, không theo số update. Batch lớn thay đổi số negative của contrastive loss; gradient accumulation batch 2 không tương đương batch thực 4 trong loss này.
- So hai trial batch 2 để xem tác động LR; so hai trial LR 1e-5 để xem tác động batch. Chưa có ô LR 5e-6/batch 4 nên đây không phải grid search đầy đủ, chưa đo interaction đầy đủ.
- FP32 giữ cố định để không lẫn tác động AMP. Nếu batch 4 OOM trên T4, ghi cấu hình không khả thi. Không tự hạ batch và gọi đó là kết quả batch 4.
- AdamW restart/LR constant khác run gốc; so các trial với nhau kiểm soát hai yếu tố này, nhưng so trial với parent chỉ cho thấy hiệu quả của cả gói fine-tune bổ sung. Muốn tách với việc train tiếp thông thường cần thêm nhánh tiếp tục OneCycle trên cùng budget.

## Screening và xác nhận

**Screening:** 64 query/category, 512 ảnh gallery/category, cùng danh sách cho mọi cấu hình; luôn chứa reference và target. Query được chọn sao cho cả reference và target không xuất hiện trong các query train gộp ba category. Gallery còn lại lấy cố định từ split val chính thức. Scoring, reference exclusion và caption processing giữ nguyên.

Gallery giảm làm bài toán dễ hơn; **không so Recall screening với Tuần 9 hoặc paper**. Script yêu cầu cờ `--allow-screening`; scope, hash query/gallery được lưu trong JSON. Tiêu chí shortlist cố định trước là trung bình của macro R@10 và R@50 với `unifashion_image_caption_score`. Bằng điểm hoặc không tốt hơn thì giữ base. Trial lỗi được ghi và loại; trial còn pending thì chưa chọn.

**Xác nhận:** chạy full 6.016 query/gallery chính thức cho parent hiện tại và trial được shortlist. Giữ cả score có/không caption trong kết quả. Chỉ cặp full-val này mới trả lời tuning có cải thiện so với cùng starting checkpoint hay không. So step 29.500 với trial bắt đầu step 60.000 sẽ trộn tác động của training thêm và tuning.

Run full-val trước trên T4 mất khoảng 62 phút cho vòng encoding/scoring; hai lượt có thể cần khoảng hai giờ cộng setup/I/O. Hai trial batch 2 xử lý 2.400 mẫu; nếu vẫn đạt benchmark 1,6 mẫu/s thì phần train khoảng 25 phút, chưa tính batch 4, screening, load model, hash và ghi checkpoint. Đây là ước lượng, không hứa thời gian.

## Chạy trên Colab

1. Chờ dòng **`Saved resumable checkpoint at step ...`**, dừng training gốc. Không chạy hai tiến trình tranh cùng GPU.
2. Upload [unifashion_tuning.zip](../checkpoints/unifashion_tuning.zip) vào `MyDrive/Thesis/UniFashion/week8/inputs`. Hai ZIP code/data gốc giữ nguyên. Bundle mới chỉ thêm code tuning/evaluation, không thay sáu source có fingerprint của trainer gốc.
3. Mở notebook mới. Kiểm tra `TRAIN_RUN` và `STUDY_NAME='fp32_lr_batch_01'`. Các lần chạy tiếp phải giữ cùng tên để dùng lại snapshot và resume trial.
4. Chạy setup/test, snapshot và prepare. `parent.json` ghi SHA-256, tiến độ và LR thực tế; giữ bản sao `parent_snapshot.pt`.
5. Chạy screening base, rồi vòng ba trial. Mỗi trial có checkpoint `trial.pt` mỗi 100 update và cuối trial. Model reload/optimizer/RNG cho resume đã kiểm thử bằng CPU fixture. Nếu ngắt ở update chưa lưu thì chạy lại từ checkpoint gần nhất.
6. Xem báo cáo screening. Chạy cell full validation khi còn quota; cell này chỉ xác nhận parent và trial shortlist. Nếu chưa đủ thời gian, có thể chạy export ngay và báo cáo đúng tiến độ.
7. Gửi **`tuning_evidence.zip`** từ thư mục `week8/tuning/<STUDY_NAME>` để hoàn thiện nhận xét. ZIP chứa config, metric, provenance, log, per-query rank và report; không chứa trọng số lớn.

Cần thêm khoảng 10–12 GB Drive cho snapshot parent, checkpoint ba trial và file tạm lúc lưu. Không upload `.pt` lên GitHub. Trong repository, logs tuần này đã có ngoại lệ `.gitignore`; các checkpoint/ZIP/file tạm bên trong vẫn bị bỏ qua.

## Khi bị ngắt hoặc có lỗi

- Quota ngắt: mở lại runtime, chạy setup với cùng `STUDY_NAME`; notebook dùng parent cũ và `--resume` cho trial đã có `trial.pt`, bỏ qua evaluation hoàn tất.
- Evaluation dở: giữ output trong thư mục có hậu tố `_interrupted_...`, chạy lại lượt đó; chưa resume từng batch evaluation.
- Nếu snapshot copy dở (chưa có `parent.json`) hoặc prepare dở (chưa có `inputs/study.json`): chọn **STUDY_NAME mới**. Không ghi đè các dữ liệu không rõ trạng thái.
- Trial thất bại trước khi có checkpoint: giữ `status.json/run.log` để chẩn đoán. Notebook không tự xóa folder. Sau khi sửa nguyên nhân, dùng study mới nếu chưa có checkpoint để resume.
- OOM batch 4: có thể báo “batch 4 FP32 không khả thi trên GPU này”; không cần làm mất mốc parent hoặc ép chạy đủ. Muốn thử FP16 thì tạo study riêng và công bố precision là biến thay đổi.
- Muốn train tiếp run gốc: dùng notebook full-training với `TRAIN_RUN/last.pt`; không dùng `trial.pt` cho trainer cũ.

## Các file phục vụ báo cáo

- Config trial: [configs/unifashion_tuning.json](../configs/unifashion_tuning.json).
- Prepare/snapshot/train/resume: [scripts/tune_unifashion.py](../scripts/tune_unifashion.py).
- Sampling và metadata: [src/unifashion_tuning.py](../src/unifashion_tuning.py).
- Recall: [scripts/evaluate_unifashion_fashioniq.py](../scripts/evaluate_unifashion_fashioniq.py).
- Report/selection: [scripts/report_unifashion_tuning.py](../scripts/report_unifashion_tuning.py).
- Kiểm thử: [tests/test_unifashion_tuning.py](../tests/test_unifashion_tuning.py).

Không dùng loss train nhỏ nhất để chọn batch, vì số negative thay đổi làm loss khó so trực tiếp. Không tự kết luận “thu hẹp khoảng cách” khi chưa có Recall full-val mới. Một seed và 1.200 ví dụ chỉ là thăm dò; ngay cả cấu hình tăng điểm val cũng chưa chứng minh cải thiện ổn định hoặc test độc lập.
