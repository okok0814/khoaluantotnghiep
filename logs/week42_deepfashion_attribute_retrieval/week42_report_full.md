# Báo cáo Tuần 42 - DeepFashion-MultiModal Attribute Retrieval

## 1. Phạm vi
- Chế độ đánh giá: `full`.
- Gallery: 41571 ảnh.
- Query: 3000.
- Checkpoint: `best_monitor.pt`.
- Giao thức: custom multi-positive attribute retrieval, không phải benchmark chính thức DeepFashion-MultiModal.

## 2. Dữ liệu
- Images thực tế: 44114.
- Captions: 42544.
- Shape rows: 42544.
- Fabric rows: 44096.
- Pattern/color rows: 44096.

## 3. Kết quả
- image_caption_score / detailed: R@1=29.80%, R@5=55.20%, R@10=65.50%, R@50=84.30% (n=1000).
- image_caption_score / fabric_pattern: R@1=36.70%, R@5=71.30%, R@10=82.80%, R@50=94.80% (n=1000).
- image_caption_score / sleeve_fabric_pattern: R@1=37.80%, R@5=62.50%, R@10=73.80%, R@50=90.30% (n=1000).
- image_score / detailed: R@1=14.40%, R@5=32.10%, R@10=41.80%, R@50=65.60% (n=1000).
- image_score / fabric_pattern: R@1=22.20%, R@5=49.20%, R@10=63.50%, R@50=85.40% (n=1000).
- image_score / sleeve_fabric_pattern: R@1=22.80%, R@5=48.20%, R@10=59.80%, R@50=81.30% (n=1000).

## 4. Ghi chú phương pháp
- Query gồm reference image + câu mô tả thuộc tính được tạo từ annotation chính thức.
- Ground truth là các ảnh cùng gender/category khớp toàn bộ thuộc tính của tier.
- Reference image bị loại khỏi ranking.
- Kết quả này không được so trực tiếp với FashionIQ Recall@K nếu không cùng dataset và protocol.

## 5. Trạng thái
- Retrieval evaluation đã chạy và đã lưu bằng chứng CSV/JSON/PNG.
