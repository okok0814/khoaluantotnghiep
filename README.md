# khoaluantotnghiep
Recommending Fashion Products Based on Images and Text Descriptions

Conference comparisons (10/11/2026): [weekly report](logs/paper_comparisons/weekly_report.md), [T4 notebook](notebooks/45_11_Conference_Comparisons.ipynb), and [run guide](docs/Conference_Comparisons_Runbook.md). Full-validation comparisons now include matched-image FashionCLIP baselines, three-seed weighted-additive/MLP controls, released CLIP4CIR/SPRC checkpoints and UniFashion initialization. Macro R@10/R@50: MLP **34.75%/56.75%**, CLIP4CIR **40.87%/64.37%**, SPRC **49.88%/71.36%**; final UniFashion **45.14%/68.53%** without gallery-caption scoring and **64.79%/80.64%** with it. The report includes verified tables and paired intervals, excludes teammate fusion results, and does not claim SOTA superiority across unequal caption resources.

Tuần 7 - UniFashion: [notebook sanity đã chạy](notebooks/39_17_UniFashion_Finetune_Sanity.ipynb).
Output notebook ghi nhận sanity thành công: 24 train queries, 12 update, checkpoint reload verified.

Tuần 8: Minh chứng nhận ngày 13/10 xác nhận snapshot FP32 ở step 29.500: hoàn tất 3 epoch và 2.500 batch trong epoch 4; chưa hoàn tất kế hoạch 10 epoch.

Đánh giá Recall@10/@50 - [notebook Colab](notebooks/41_15_UniFashion_FashionIQ_Recall_Evaluation.ipynb).
Đã nhận và kiểm tra kết quả đủ 6.016 val queries. Macro R@10/R@50: FashionCLIP additive 23,62%/41,54%; UniFashion không cộng score caption gallery 36,21%/60,41%; có cộng score caption gallery 57,08%/75,43%.

Tuần 10 (20/10/2026) - tinh chỉnh learning rate/batch size, [notebook Colab đã chạy](notebooks/42_14_UniFashion_Hyperparameter_Tuning.ipynb).
Đã hoàn thành ba trial và xác nhận full validation. Cập nhật 27/10: base step 60.000 đạt macro R@10/R@50 **62,57%/79,18%**, tốt hơn trial LR 1e-5/batch 4 (**61,88%/78,88%**). Giữ base theo mean macro Recall. [Báo cáo và minh chứng](logs/week42_unifashion_tuning/full_validation_report.md).

Tuần 43–44 — Cập nhật 02/11/2026: run FP32 gốc **hoàn tất 10 epoch / 90.000 update**. Đã kiểm tra đủ 6.016 validation query: step90000 đạt macro R@10/R@50 **64,79%/80,64%**, cao hơn step60000 **2,21/1,46 điểm phần trăm**. Đã nhận receipt lưu `best_recall_step90000.pt`, xác minh checksum và CPU reload tại Colab. Đây là best trong hai ứng viên step60000/step90000, chưa phải so sánh mọi epoch. [Kết quả kiểm tra](logs/week43_44_unifashion_completion/evidence_import.json).

[Notebook chốt checkpoint và so sánh](notebooks/44_12_UniFashion_Completion_Comparison.ipynb) đã có minh chứng hoàn tất phần UniFashion. FashionCLIP additive đạt **23,62%/41,54%**; UniFashion step90000 bỏ score caption gallery đạt **45,14%/68,53%**. Attention-fusion vẫn chờ kết quả cùng giao thức. Kết quả hiện là **validation**; 6.118 query test local thiếu target nên chưa thể báo test Recall. Không cần chạy lại training/evaluation UniFashion cho báo cáo này.

