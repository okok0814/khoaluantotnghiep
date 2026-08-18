# Phân tích Kiến trúc các Mô hình Vision-Language cho Thời trang (VLP)
**Dự án:** Hệ thống gợi ý sản phẩm thời trang bằng hình ảnh và văn bản (Bachelor's Thesis)
**Mục tiêu tuần 1:** Đọc toàn bộ 10 bài báo trong danh sách của GVHD; chú giải sơ đồ phương pháp của 6 kiến trúc vision-language (FashionVLP, FashionViL, FashionSAP, FAME-ViL, FashionERN, UniFashion).

## 1. FashionVLP (CVPR 2022) - Truy xuất với Phản hồi (Retrieval with Feedback)
*   **Luồng đặc trưng thị giác (Visual Flow):** Không chỉ trích xuất ảnh tổng thể (global image), mô hình còn trích xuất các đặc trưng cục bộ (crop) và các điểm mốc thời trang (fashion landmarks như cổ áo, gấu áo).
*   **Cơ chế dung hợp:** Sử dụng mạng Asymmetric Attention (Chú ý bất đối xứng). Thay vì gộp chung text và image ngay từ đầu, nó dùng text (câu phản hồi của người dùng) để hướng dẫn sự chú ý lên các vùng hình ảnh cụ thể, giúp giữ nguyên cấu trúc không gian của ảnh.

## 2. FashionViL (ECCV 2022) - Học biểu diễn V+L tập trung vào thời trang
*   **Luồng dữ liệu:** Đưa ra một mô hình Transformer thống nhất để xử lý cả hai luồng.
*   **Cơ chế dung hợp:** Là kiến trúc Cross-Attention (Dung hợp sâu). Hình ảnh được chia thành các patch (như ViT) và kết hợp với token của văn bản, trải qua nhiều lớp Transformer để học các mối liên hệ chéo.
*   **Pre-training Tasks:** Khớp ảnh-chữ (Image-Text Matching), Khôi phục từ bị che (Masked Language Modeling) và Khôi phục vùng ảnh bị che (Masked Region Modeling).

## 3. FashionSAP (CVPR 2023) - Symbols & Attributes Prompt (Mô hình thực nghiệm chính)
*   **Đặc điểm nổi bật:** Sử dụng các "Ký hiệu" (Symbols) và "Thuộc tính" (Attributes) chi tiết làm Prompt. Khắc phục nhược điểm của các mô hình trước thường bỏ qua chi tiết nhỏ (ví dụ: họa tiết sọc ngang, cổ chữ V).
*   **Cơ chế dung hợp:** Kết hợp Fine-grained Text (mô tả thuộc tính chi tiết) với đặc trưng ảnh ở cấp độ patch.
*   **Mức độ tích hợp Khóa luận:** Đây là **repository được chọn để tái hiện (hssip/FashionSAP)**. Cấu trúc tách biệt rõ ràng các file pre-train và downstream (retrieval, category recognition) rất lý tưởng để chạy thực nghiệm định lượng trên bộ dữ liệu FashionIQ.

## 4. FAME-ViL (CVPR 2023) - Multi-Tasking cho tác vụ không đồng nhất
*   **Mục tiêu:** Xử lý cùng lúc nhiều tác vụ (Heterogeneous tasks) thay vì đào tạo riêng lẻ từng mô hình.
*   **Cơ chế dung hợp:** Sử dụng các "Task-specific Adapters" (Bộ chuyển đổi chuyên biệt) gắn vào một backbone Vision-Language dùng chung. Nó giúp mô hình vừa làm tốt việc tìm kiếm (Retrieval) vừa làm tốt việc nhận dạng thuộc tính (Attribute Recognition) mà không bị "quên" kiến thức.

## 5. FashionERN (AAAI 2024) - Enhance-and-Refine Network
*   **Vấn đề giải quyết:** Trong truy xuất cấu thành (Composed Image Retrieval - CIR), văn bản phản hồi thường ngắn và khó làm biến đổi vector hình ảnh tham chiếu một cách hiệu quả.
*   **Cơ chế dung hợp:** Chia làm 2 giai đoạn:
    1.  *Enhance (Tăng cường):* Làm giàu đặc trưng của văn bản phản hồi bằng cách khai thác thông tin từ ảnh gốc.
    2.  *Refine (Tinh chỉnh):* Dùng văn bản đã được tăng cường để tinh chỉnh (nhấn mạnh hoặc loại bỏ) các đặc trưng trong vector hình ảnh đích.

## 6. UniFashion (EMNLP 2024) - Mô hình Thống nhất (Retrieval & Generation)
*   **Kiến trúc:** Một mô hình seq2seq (Sequence-to-Sequence) đa phương thức, hợp nhất cả bài toán hiểu (Retrieval) và bài toán sinh (Generation / Captioning).
*   **Cơ chế dung hợp:** Gắn các vector ảnh như một chuỗi token đầu vào, đặt cạnh các token văn bản, và tối ưu hóa bằng hàm mất mát sinh văn bản (Generative Loss). Điều này mang hơi hướng của các mô hình LLM đa phương thức (Large Multimodal Models) hiện đại.