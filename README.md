# khoaluantotnghiep
Recommending Fashion Products Based on Images and Text Descriptions

Tuần 7 — UniFashion: [hướng dẫn](docs/Week7_UniFashion_Sanity.md),
[notebook sanity đã chạy](notebooks/39_17_UniFashion_Finetune_Sanity.ipynb).
Output notebook ghi nhận sanity thành công: 24 train queries, 12 update, checkpoint reload verified.

Tuần 8 — [fine-tune full FashionIQ và theo dõi loss](docs/Week8_UniFashion_Finetune.md),
[báo cáo tiến độ](reports/week8_unifashion/weekly_report.md).
Minh chứng nhận ngày 13/10 xác nhận snapshot FP32 ở step 29.500: hoàn tất 3 epoch và 2.500 batch trong epoch 4; chưa hoàn tất kế hoạch 10 epoch.

Đánh giá Recall@10/@50 — [hướng dẫn](docs/FashionIQ_Recall_Evaluation.md),
[notebook Colab](notebooks/41_01_UniFashion_FashionIQ_Recall_Evaluation.ipynb),
[bảng baseline và báo cáo](reports/week9_fashioniq_evaluation/evaluation_report.md).
Đã nhận và kiểm tra kết quả đủ 6.016 val queries. Macro R@10/R@50: FashionCLIP additive 23,62%/41,54%; UniFashion không cộng score caption gallery 36,21%/60,41%; có cộng score caption gallery 57,08%/75,43%.
[Báo cáo tuần và phân tích kết quả](reports/week9_fashioniq_evaluation/weekly_report.md).
