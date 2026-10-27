# khoaluantotnghiep
Recommending Fashion Products Based on Images and Text Descriptions

Tuần 7 - UniFashion: [notebook sanity đã chạy](notebooks/39_17_UniFashion_Finetune_Sanity.ipynb).
Output notebook ghi nhận sanity thành công: 24 train queries, 12 update, checkpoint reload verified.

Tuần 8: Minh chứng nhận ngày 13/10 xác nhận snapshot FP32 ở step 29.500: hoàn tất 3 epoch và 2.500 batch trong epoch 4; chưa hoàn tất kế hoạch 10 epoch.

Đánh giá Recall@10/@50 - [notebook Colab](notebooks/41_15_UniFashion_FashionIQ_Recall_Evaluation.ipynb).
Đã nhận và kiểm tra kết quả đủ 6.016 val queries. Macro R@10/R@50: FashionCLIP additive 23,62%/41,54%; UniFashion không cộng score caption gallery 36,21%/60,41%; có cộng score caption gallery 57,08%/75,43%.

Tuần 10 (20/10/2026) - tinh chỉnh learning rate/batch size, [notebook Colab đã chạy](notebooks/42_14_UniFashion_Hyperparameter_Tuning.ipynb).
Đã hoàn thành ba trial và xác nhận full validation. Cập nhật 27/10: base step 60.000 đạt macro R@10/R@50 **62,57%/79,18%**, tốt hơn trial LR 1e-5/batch 4 (**61,88%/78,88%**). Giữ base theo mean macro Recall. [Báo cáo và minh chứng](logs/week42_unifashion_tuning/full_validation_report.md).

Tuần 43 — hoàn tất training và lưu checkpoint tốt nhất. Hiện đang ở epoch 9, step 78.000; chưa xác nhận hoàn tất 10 epoch. Snapshot step 60.000 là tốt nhất trong các checkpoint đã đánh giá, cần bảo toàn trước khi so với snapshot cuối.

