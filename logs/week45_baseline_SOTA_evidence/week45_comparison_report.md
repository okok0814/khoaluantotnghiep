# Đối chiếu UniFashion, baseline và paper gốc — tổng hợp Tuần 44–45

- Trạng thái: evaluation_completed.
- Mô hình đề xuất: `best_monitor.pt`, epoch 8, có 8 khóa Attention-Gated Fusion.
- FashionIQ validation: 6016 queries.
- Macro Recall@10: **64.285%**.
- Macro Recall@50: **80.739%**.
- Bảng baseline lịch sử: `measured_recall_comparison.csv`. Đây chưa phải ablation cùng điều kiện.
- Bảng số liệu công bố: `paper_published_results.csv` (UniFashion 2024, Table 4, p.8).
- Giao thức FashionIQ: xem `protocol_reconciliation.json`; mọi khác biệt định dạng đều đã được kiểm chứng và ghi lại.
- Không dùng monitor loss thay Recall@K.
- Chưa đủ cơ sở để khẳng định mức đóng góp riêng của fusion nếu thiếu baseline UniFashion không fusion trong cùng điều kiện.
- Chưa xác minh bảng xếp hạng SOTA hiện hành hoặc kết quả MAPNet/SPN4CIR, nên không đưa số liệu giả.

## Minh chứng notebook
1. Cell mã Mục 3: checkpoint audit (epoch, tham số fusion, SHA-256).
2. Cell mã Mục 4: log đánh giá hoàn tất và số lượng query.
3. Cell mã Mục 5: Recall@K, biểu đồ so sánh.
4. Cell mã Mục 6: bảng số liệu công bố và nguồn paper.
