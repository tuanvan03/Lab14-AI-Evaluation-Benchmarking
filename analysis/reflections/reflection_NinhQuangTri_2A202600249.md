# Báo cáo Cá nhân — Lab 14: AI Evaluation & Benchmarking

**Họ tên:** Ninh Quang Trí
**MSSV:** 2A202600249
**Branch:** `main`

---

## 1. Engineering Contribution (15 điểm)

### 1.1 Các module đã đóng góp

#### LLM-as-a-Judge Engine (`engine/llm_judge.py`)

Tôi đã thiết kế và triển khai hệ thống chấm điểm tự động sử dụng LLM để thay thế đánh giá thủ công:

- **Class `LLMJudge`**: Xây dựng bộ khung chấm điểm dựa trên rubrics cứng cho `Accuracy` và `Tone` (thang điểm 1-5). Sử dụng `AsyncOpenAI` để tối ưu hóa hiệu năng gọi API.
- **`evaluate_multi_judge()`**: Triển khai logic **Dual-Judge**. Hệ thống gọi song song hai model (`gpt-4o` và `gpt-4o-mini`) bằng `asyncio.gather` để lấy kết quả khách quan nhất.
- **Conflict Resolution**: Thiết kế logic xử lý khi hai judge bất đồng ý kiến. Tính toán `agreement_rate` và áp dụng **Strict Rule** (lấy điểm thấp nhất) nếu mức độ lệch điểm > 1, đảm bảo tính khắt khe của hệ thống đánh giá.
- **Structured Output**: Ép kiểu output của LLM về JSON bằng `response_format={"type": "json_object"}` để đảm bảo pipeline ổn định, không bị crash bởi text thừa của LLM.

#### Retrieval Evaluation Metrics (`engine/retrieval_eval.py`)

Hoàn thiện công cụ đánh giá giai đoạn Retrieval cho hệ thống RAG:

- **`RetrievalEvaluator`**: Triển khai các hàm toán học tính toán **Hit Rate** (có tìm thấy tài liệu đúng trong top-K không) và **MRR** (Mean Reciprocal Rank - tính đến vị trí của tài liệu đúng).
- **Batch Evaluation**: Xây dựng pipeline `evaluate_batch` tự động load `golden_set.jsonl`, thực hiện query vào ChromaDB và tính toán các chỉ số trung bình.
- **Reporting System**: Viết hàm `print_report` thống kê hiệu năng theo độ khó (easy/medium/hard) và theo loại query (story/lesson), giúp đội ngũ nhận diện được các "weak spot" của hệ thống.

#### Benchmark Runner & Cost Tracker (`main.py`)

Phát triển bộ điều phối toàn bộ quá trình benchmarking:

- **`ExpertEvaluator`**: Tích hợp metrics từ `RetrievalEvaluator` vào flow đánh giá chung.
- **Cost Calculation**: Triển khai hàm `calculate_costs` tính toán chi phí token (USD) dựa trên bảng giá thực tế của gpt-4o và gpt-4o-mini, giúp quản lý ngân sách khi eval số lượng lớn.
- **Parallel vs Sequential**: Xây dựng benchmark runner hỗ trợ cả hai chế độ chạy tuần tự và song song, từ đó chứng minh được speedup factor (~3-4x) khi dùng `asyncio`.
- **Automated Decision**: Thêm logic so sánh giữa phiên bản Base (V1) và Optimized (V2) để đưa ra quyết định **Approve/Reject** dựa trên sự thay đổi của score và chi phí.

### 1.2 Git Commits tiêu biểu

| Commit    | Mô tả                                                                                               |
| --------- | --------------------------------------------------------------------------------------------------- |
| `1177b5e` | `feat: implement LLM-based multi-model judge and retrieval evaluation metrics for RAG benchmarking` |
| `aca8ff7` | `feat: implement multi-model LLM judge and benchmark runner with cost tracking`                     |
| `1c8a76b` | `initialize main_agent.py with core agent logic and execution flow`                                 |

---

## 2. Technical Depth (15 điểm)

### 2.1 Mean Reciprocal Rank (MRR)

MRR là một metric quan trọng trong Retrieval vì nó không chỉ quan tâm đến việc "có tìm thấy không" (như Hit Rate) mà còn quan tâm đến **thứ hạng** của kết quả đúng:
- Công thức: `1 / Rank`. Nếu tài liệu đúng ở vị trí số 1 -> RR=1.0. Nếu ở vị trí số 10 -> RR=0.1.
- Ý nghĩa: Giúp tối ưu hóa hệ thống để đẩy các kết quả quan trọng nhất lên đầu, giảm thiểu việc LLM phải đọc quá nhiều context nhiễu (noise) trong RAG.

### 2.2 LLM-as-a-Judge & Agreement Rate

Khi dùng LLM để chấm điểm, thách thức lớn nhất là tính nhất quán (consistency):
- **Agreement Rate**: Chúng tôi đo mức độ đồng thuận giữa hai model khác nhau. Nếu `agreement_rate` cao, kết quả đánh giá đáng tin cậy. 
- **Strict Rule**: Trong các case nhạy cảm, việc lấy điểm trung bình có thể che lấp các lỗi nghiêm trọng. Tôi áp dụng quy tắc lấy điểm tối thiểu để bắt buộc Agent phải hoàn thiện ở mức cao nhất.

### 2.3 Trade-off giữa Chi phí và Chất lượng Benchmarking

Sử dụng `gpt-4o` làm judge cho độ chính xác cực cao nhưng chi phí đắt hơn ~50 lần so với `gpt-4o-mini`:
- Chiến lược của tôi: Dùng `gpt-4o-mini` cho các test case đơn giản và dùng dual-judge kết hợp `gpt-4o` cho các case phức tạp/Adversarial.
- Kết quả: Giảm được đáng kể chi phí tổng quát mà vẫn duy trì được "gold quality" cho những case quan trọng nhất.

---

## 3. Problem Solving (10 điểm)

### Vấn đề 1: LLM Judge trả về JSON không hợp lệ hoặc text thừa

- **Triệu chứng**: Khi không có ràng buộc chặt chẽ, LLM thỉnh thoảng thêm giải thích dông dài hoặc markdown code block làm hàm `json.loads()` bị fail.
- **Root Cause**: LLM tự do tạo format output.
- **Giải pháp**: Thiết lập `response_format={"type": "json_object"}` trong tham số API call và yêu cầu rõ ràng trong system prompt. Thêm khối `try-except` để parse thủ công hoặc trả về giá trị default (3-3) nếu vẫn fail.

### Vấn đề 2: Tốc độ Benchmark quá chậm khi chạy Sequential

- **Triệu chứng**: Chạy 50 test case mất hơn 2 phút nếu gọi từng cái một.
- **Giải pháp**: Chuyển đổi toàn bộ logic sang `async/await`. Sử dụng `asyncio.gather()` để gọi đồng thời nhiều request. 
- **Kết quả**: Thời gian benchmark giảm từ 120s xuống còn ~35s (speedup ~3.4x), cho phép developer feedback loop nhanh hơn.

### Vấn đề 3: Sai lệch chỉ số MRR do metadata trong VectorDB

- **Triệu chứng**: Hit Rate cao nhưng MRR thấp, tài liệu đúng thường xuất hiện ở cuối trang.
- **Giải pháp**: Phát hiện ra do việc đánh index ID không nhất quán giữa giai đoạn Ingestion và Retrieval. Tôi đã đồng bộ lại format ID `{filename}_chunk_{idx}` để hàm `calculate_mrr` có thể so sánh chính xác với `ground_truth_ids`.
