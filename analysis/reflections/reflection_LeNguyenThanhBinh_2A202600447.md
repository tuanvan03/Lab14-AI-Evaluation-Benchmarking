# Báo cáo Cá nhân — Lab 14: AI Evaluation & Benchmarking  

**Họ tên:** Lê Nguyễn Thanh Bình  
**MSSV:** 2A202600447

---

## Engineering Contribution

Trong quá trình thực hiện Lab 14, tôi chủ yếu tập trung vào việc tìm hiểu cách xây dựng một hệ thống đánh giá cho các ứng dụng AI, đặc biệt là các hệ thống RAG (Retrieval-Augmented Generation).

Tôi đã tiếp cận cách tổ chức một agent với lịch sử hội thoại (conversation history) và hiểu được vai trò của việc rewrite lại câu hỏi trước khi truy vấn tài liệu. Điều này giúp cải thiện độ chính xác của retrieval khi câu hỏi ban đầu chưa rõ ràng hoặc thiếu ngữ cảnh.

Ngoài ra, tôi cũng tìm hiểu cách sinh dữ liệu kiểm thử (synthetic data) bằng LLM. Việc phân loại các dạng câu hỏi như:

- **Single-turn**: fact-check, adversarial, ambiguous, edge cases  
- **Multi-turn**: carry-over, correction  

giúp tôi hiểu rõ hơn về cách thiết kế bộ test đa dạng để đánh giá toàn diện hệ thống.

Bên cạnh đó, tôi cũng quan sát và học hỏi cách sử dụng nhiều model làm judge để đánh giá câu trả lời, cũng như các vấn đề phát sinh khi dùng model nhỏ cho tác vụ này.

---

## Technical Depth

### MRR (Mean Reciprocal Rank)

MRR là một metric quan trọng để đánh giá hiệu quả của hệ thống retrieval. Nó đo lường vị trí xuất hiện của tài liệu đúng trong danh sách kết quả.

- Nếu **MRR ≈ 1** → hệ thống trả kết quả đúng ở vị trí đầu  
- Nếu **MRR thấp** → kết quả đúng xuất hiện nhưng thứ hạng chưa tốt  

Công thức: MRR = (1/N) × Σ (1/rank_i)

Trong đó:
- `N`: số lượng truy vấn  
- `rank_i`: vị trí của tài liệu đúng ở truy vấn thứ i  

---

### Cohen’s Kappa

Cohen’s Kappa giúp đo độ đồng thuận giữa các judge, đồng thời loại bỏ yếu tố ngẫu nhiên.

Công thức: k = (P_observed - P_expected) / (1 - P_expected)

Trong đó:
- `P_observed`: mức độ đồng thuận quan sát được  
- `P_expected`: mức độ đồng thuận do ngẫu nhiên  

Ý nghĩa:
- `k ≈ 0`: đồng thuận do may mắn  
- `k = 0.4 – 0.6`: đồng thuận vừa phải  
- `k > 0.6`: đồng thuận tốt  

---

### Position Bias

Position Bias là hiện tượng LLM có xu hướng chọn câu trả lời dựa trên vị trí thay vì chất lượng.

Ví dụ:
- Luôn chọn đáp án đầu tiên  
- Hoặc bị ảnh hưởng bởi thứ tự trình bày  

Nguyên nhân:
- LLM học từ dữ liệu văn bản của con người  
- Con người thường đặt thông tin quan trọng ở đầu hoặc cuối  

Hệ quả:
- Benchmark bị sai lệch  
- Model ở vị trí thuận lợi sẽ có lợi thế không công bằng  

Cách xử lý:
- Shuffle thứ tự câu trả lời  
- Đánh giá hai chiều (swap vị trí)  
- Chỉ giữ kết quả nhất quán  

---

### Trade-off giữa Chi phí và Chất lượng

Trong thực tế, luôn tồn tại sự đánh đổi giữa chi phí và chất lượng model:

- **Model nhỏ (ví dụ: Haiku)**  
  - Nhanh, rẻ  
  - Nhưng kém trong các task phức tạp  

- **Model trung bình (ví dụ: Sonnet)**  
  - Cân bằng tốt giữa chi phí và hiệu năng  
  - Phù hợp cho production  

- **Model lớn (ví dụ: Opus)**  
  - Chất lượng cao nhất  
  - Nhưng chi phí đắt và chậm  

=> Cần lựa chọn model phù hợp với mục tiêu sử dụng thay vì luôn chọn model mạnh nhất.

---

## Problem Solving

Trong quá trình làm lab, tôi nhận ra tầm quan trọng của việc xử lý ngữ cảnh trong các bài toán multi-turn.

Cụ thể:
- Nếu không truyền đầy đủ **conversation history**  
- Hệ thống sẽ không hiểu được ngữ cảnh trước đó  
→ dẫn đến câu trả lời sai hoặc thiếu thông tin  

Bài học rút ra:
- Context là yếu tố bắt buộc trong hệ thống hội thoại  
- Cần đảm bảo được truyền xuyên suốt trong pipeline  

---

## Tổng kết

Qua Lab 14, tôi rút ra một số điểm quan trọng:

- Dataset đánh giá quan trọng không kém model  
- LLM-as-a-judge cần được kiểm soát kỹ  
- Benchmark nếu làm không cẩn thận sẽ dẫn đến kết luận sai  

Lab giúp tôi hiểu rõ hơn về cách đánh giá hệ thống AI trong thực tế, nơi mà độ tin cậy và tính nhất quán quan trọng không kém hiệu năng.