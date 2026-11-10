# Báo cáo tuần — Tinh chỉnh hyperparameter UniFashion

Cập nhật **20/10/2026** từ output đã lưu của [notebook Colab 42_14](../../notebooks/42_14_UniFashion_Hyperparameter_Tuning.ipynb).

**Đã hoàn thành huấn luyện ba trial và screening của cả ba trial cùng base. Full validation bị ngắt do hết quota Colab theo thông tin người chạy; chưa đủ kết quả để xác nhận cải thiện trên toàn bộ val hoặc thu hẹp khoảng cách với paper.**

Nhiệm vụ: tinh chỉnh learning rate và batch size cho mô hình đã fine-tune để thu hẹp khoảng cách với số liệu công bố, nếu khả thi.

## Minh chứng và thiết lập

Notebook ghi GPU **Tesla T4**, precision **FP32**, seed **42**, study `fp32_lr_batch_01`. Output snapshot xác nhận parent tại **step 60.000**, tương ứng **6 epoch hoàn tất và 6.000 batch trong epoch 7**; chưa hoàn tất kế hoạch 10 epoch.

- SHA-256 parent được ghi trong output: `df8226a0e61e41ebb5bf88f1870b0176118f75d14612ceb8c0e04cf91309a683`.
- LR kế tiếp của OneCycle trong run gốc: khoảng **6,6761e-6**; 2e-5 là max LR cấu hình, không phải LR thực tế tại mọi bước.
- Mỗi trial bắt đầu độc lập từ cùng parent, tạo AdamW mới, LR cố định; giữ frozen EVA-G và preprocessing. Không thay checkpoint/optimizer của run gốc.
- Tập train con: **1.200 ví dụ**, 400/category, cùng danh sách và thứ tự. Ngân sách bằng nhau theo số ví dụ, không theo số update; batch lớn thay đổi số negative.
- Screening: **192 query**, 64/category; gallery **512 ảnh/category**, loại reference. Có 1.536 vị trí gallery theo category nhưng 1.534 ảnh duy nhất cần kiểm tra do overlap category.
- Output setup ghi **12 tests passed**. Các test là kiểm thử phần mềm, không thay thế đánh giá retrieval.

Đã trích output và kiểm tra lại phép tính macro/micro, số update hoàn tất, LR/batch và tiêu chí chọn. Nguồn kiểm tra: [notebook_output_audit.json](notebook_evidence/notebook_output_audit.json). Đây là minh chứng từ notebook, **chưa nhận ZIP chứa JSON kết quả gốc, CSV rank từng query hoặc checkpoint** để kiểm tra độc lập các thứ hạng. Không thực thi lại các cell của notebook khi trích xuất.

## Huấn luyện đã hoàn thành

| Trial | LR cố định | Batch FP32 | Ví dụ đã xử lý | Update đã lưu | Trạng thái |
|---|---:|---:|---:|---:|---|
| Base step 60.000 | Không train thêm | — | 0 | 0 | Screening hoàn tất |
| `lr1e5_b2` | 1e-5 | 2 | 1.200 | 600/600 | Train và screening hoàn tất |
| `lr5e6_b2` | 5e-6 | 2 | 1.200 | 600/600 | Train và screening hoàn tất |
| `lr1e5_b4` | 1e-5 | 4 | 1.200 | 300/300 | Train và screening hoàn tất |

Cả ba trial có dòng lưu checkpoint cuối và báo cáo `trial_completed`. Batch 4 FP32 đã chạy thành công trong lượt này; không có trial thất bại trong `selection`. Không cần train lại ba trial để tiếp tục full validation nếu các file trên Drive còn nguyên.

## Kết quả screening

**Bảng sau chỉ dành cho gallery giảm, không so trực tiếp với Recall full-val Tuần 9 hoặc số liệu paper.** Tiêu chí chọn cố định trước: trung bình của macro Recall@10 và Recall@50 với `unifashion_image_caption_score`.

| Cấu hình | R@10 (%) | R@50 (%) | Mean R (%) | Δ R@10 (pp) | Δ R@50 (pp) | Δ Mean R (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Base step 60.000 | 88,02 | 96,35 | 92,19 | 0,00 | 0,00 | 0,00 |
| LR 1e-5, batch 2 | 87,50 | 95,83 | 91,67 | −0,52 | −0,52 | −0,52 |
| LR 5e-6, batch 2 | 87,50 | 96,35 | 91,93 | −0,52 | 0,00 | −0,26 |
| **LR 1e-5, batch 4** | **90,10** | **95,31** | **92,71** | **+2,08** | **−1,04** | **+0,52** |

![Screening trên gallery giảm](screening_comparison.png)

`lr1e5_b4` được đưa vào shortlist vì có Mean R cao nhất, **không phải vì cải thiện cả hai metric**. Với 192 query chia đều ba category, số hit suy ra từ các tỷ lệ là:

- Base: **169/192** đúng top-10, **185/192** đúng top-50.
- Batch 4: **173/192** đúng top-10, **183/192** đúng top-50.

Chênh lệch ròng là **+4 hit top-10 và −2 hit top-50**; đây không phải danh sách query cải thiện/suy giảm, vì chưa có rank từng query để ghép cặp. Mức tăng Mean R chỉ **0,52 pp** trên tập nhỏ, chưa chứng minh cải thiện ổn định hoặc có ý nghĩa thống kê.

| Category | Base R@10 / R@50 (%) | Batch 4 R@10 / R@50 (%) |
|---|---:|---:|
| dress | 78,13 / 93,75 | 85,94 / 92,19 |
| shirt | 95,31 / 98,44 | 95,31 / 98,44 |
| toptee | 90,63 / 96,88 | 89,06 / 95,31 |

Tăng top-10 tập trung ở dress, shirt giữ nguyên, toptee giảm. Không kết luận cấu hình cải thiện đồng đều mọi category.

CSV đầy đủ: [screening_comparison.csv](notebook_evidence/screening_comparison.csv), [screening_all_categories.csv](notebook_evidence/screening_all_categories.csv).

## Kiểm tra biến thể không cộng score caption gallery

Hai biến thể đều dùng ảnh reference và mô tả điều chỉnh; “không caption gallery” chỉ bỏ thành phần score bổ sung của caption gallery.

| Cấu hình | R@10 (%) | R@50 (%) |
|---|---:|---:|
| Base | 75,52 | 92,19 |
| LR 1e-5, batch 2 | 76,04 | 92,19 |
| LR 5e-6, batch 2 | 75,52 | 92,71 |
| LR 1e-5, batch 4 | 71,88 | 92,71 |

Batch 4 giảm **3,65 pp R@10** khi không cộng caption gallery. Kết quả shortlist vì vậy phụ thuộc cách scoring đã chọn trước; chưa thể nói biểu diễn ảnh/mô tả tự nó tốt hơn sau tuning.

## Loss được ghi trong output

[logged_training_updates.csv](notebook_evidence/logged_training_updates.csv) chỉ chứa các update được in ra (update 1 và mỗi 25 update), **không phải toàn bộ lịch sử loss**. Không dùng trung bình các dòng này làm train loss toàn trial.

Ở update 300 cuối trial batch 4, loss in ra là **8,59095**, gradient norm **trước clipping** là **849,271**, `amp_retries=0`; pipeline cấu hình clipping tại 1,0. Đây là một batch có loss/gradient tăng cao cần theo dõi, không đủ kết luận toàn run phân kỳ. Việc lưu checkpoint và screening hoàn tất cũng không đủ chứng minh hội tụ. Không chọn batch theo raw train loss vì số negative khác nhau.

## Phần chưa hoàn thành

Cell code thứ 6, “Inspect screening and confirm on full validation”, đã bắt đầu **full validation của base**, chưa tới full validation của trial được chọn. Output cuối được lưu:

```text
dress: gallery 3817/3817
...
dress: queries 900/2017
```

Đây là dòng tiến độ cuối nhìn thấy, không bảo đảm runtime thực tế dừng chính xác tại query 900. Chưa có summary full-val của base hoặc trial; chưa có bằng chứng hoàn tất dress/shirt/toptee cho lượt full validation này. Việc hết quota do người chạy xác nhận; notebook không có thông báo lỗi quota trong output.

**Không lấy 90,10% / 95,31% screening để tuyên bố vượt paper hoặc cải thiện so với 57,08% / 75,43% full-val ở step 29.500.** Số query/gallery và parent checkpoint khác nhau. Cần full-val base step 60.000 và full-val trial từ cùng parent trước khi kết luận hiệu quả tuning trên giao thức dự án; so paper còn phải nêu các khác biệt trong [hướng dẫn](../../docs/Week10_UniFashion_Tuning.md).

## Đoạn kết quả có thể đưa vào báo cáo tuần

> Trong tuần này, nhóm đã thực hiện ba thí nghiệm tinh chỉnh learning rate và batch size cho UniFashion, cùng khởi tạo từ checkpoint FP32 tại step 60.000. Mỗi cấu hình được huấn luyện bổ sung trên cùng 1.200 ví dụ và đánh giá screening trên 192 truy vấn với gallery 512 ảnh cho mỗi nhóm sản phẩm. Cấu hình learning rate 1e-5, batch size 4 đạt macro Recall@10 là 90,10% và Recall@50 là 95,31%; trung bình hai metric đạt 92,71%, cao hơn base 0,52 điểm phần trăm. Tuy nhiên, Recall@50 giảm 1,04 điểm phần trăm, nên cấu hình này mới được chọn làm ứng viên kiểm chứng tiếp theo. Cả ba trial đã hoàn tất huấn luyện và screening. Full validation bị gián đoạn do giới hạn sử dụng Colab, vì vậy chưa đủ cơ sở kết luận cải thiện trên toàn bộ FashionIQ validation hoặc thu hẹp khoảng cách với số liệu công bố.

## Tiếp tục sau khi có quota

1. Mở lại notebook và chạy **ba cell code đầu** để mount Drive, setup môi trường và khôi phục biến/hàm. Giữ `STUDY_NAME='fp32_lr_batch_01'`; cell snapshot dùng lại parent đã lưu.
2. Chạy trực tiếp **cell code thứ 6** “Inspect screening and confirm on full validation”. Không cần chạy lại cell train ba trial. Nếu chạy lại các cell screening/train, notebook cũng bỏ qua các lượt đã có trạng thái hoàn tất.
3. Lượt full-val base chưa hoàn tất sẽ được giữ trong folder có hậu tố `_interrupted_...` rồi **chạy lại từ đầu**, không resume từ query 900. Sau base, notebook chạy full-val `lr1e5_b4`.
4. Chạy cell export để tạo `tuning_evidence.zip`; gửi ZIP để kiểm tra JSON/per-query và hoàn thiện kết luận. Nếu chỉ cần báo cáo tiến độ, có thể export ngay sau setup mà chưa chạy lại full validation; không cần GPU cho riêng việc đóng gói.

Dữ liệu dự kiến còn trên Drive: `MyDrive/Thesis/UniFashion/week8/tuning/fp32_lr_batch_01/`, gồm `parent_snapshot.pt`, `inputs`, `trials/*/trial.pt`, `screen/*`, `report/selection.json`. Chưa kiểm tra trực tiếp Drive trong lượt đọc notebook này.

Tạo lại minh chứng từ notebook đã lưu:

```powershell
.venv/Scripts/python.exe -B scripts/extract_unifashion_tuning_notebook.py --notebook notebooks/42_14_UniFashion_Hyperparameter_Tuning.ipynb --output logs/week10_unifashion_tuning/notebook_evidence
```
