# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 54
- **Tỉ lệ Pass/Fail:** 38 Pass / 16 Fail (Dựa trên ngưỡng điểm Judge < 3)
- **Chỉ số Retrieval:**
    - Hit Rate: 40.7%
    - Mean Reciprocal Rank (MRR): 0.33
- **Điểm LLM-Judge trung bình:** 3.03 / 5.0
- **Độ đồng thuận (Agreement):** 92.4% (Rất cao)

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| Retrieval Failure | 12 | Vector DB không trả về đúng chunk chứa đáp án (Hit rate thấp). |
| Hallucination | 3 | Agent tự bịa ra thông tin khi không tìm thấy trong context. |
| Reasoning Error | 1 | Agent tìm thấy context nhưng diễn giải sai mâu thuẫn giữa các dị bản. |

## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất)

### Case #1: Thất bại truy vấn "Anh ấy đã lấy gì..."
1. **Symptom:** Agent trả lời sai hoàn toàn về vật dụng (giày da bò thay vì đáp án đúng).
2. **Why 1:** LLM nhận được context không liên quan từ Retrieval.
3. **Why 2:** Query rewrite biến câu "Anh ấy..." thành một câu quá mơ hồ hoặc sai thực thể.
4. **Why 3:** Hệ thống Embedding (OpenAI) không phân biệt tốt được các đại từ "anh ấy" trong các câu chuyện khác nhau.
5. **Why 4:** Thủ tục Chunking chia nhỏ các đoạn hội thoại làm mất đi chủ ngữ của đoạn văn.
6. **Root Cause:** Thiếu cơ chế đánh dấu thực thể (Entity Linking) và Chunking làm mất Context (Context Window Loss).

### Case #187: Hành động của chị dâu Ngưu Lang
1. **Symptom:** Agent trả lời chị dâu "hãm hại" Ngưu Lang trong khi đáp án chuẩn khác.
2. **Why 1:** Context trả về không chứa đoạn văn về hành động cụ thể của chị dâu.
3. **Why 2:** Điểm Similarity của các chunk chứa đáp án thấp hơn các chunk chứa từ khóa "chị dâu" ở các đoạn khác.
4. **Why 3:** Indexing chỉ dựa trên Keyword mà chưa có Reranking để ưu tiên độ liên quan.
5. **Root Cause:** Thiếu bước Re-ranking sau khi Retrieval.

### Case #302: Phân tích mâu thuẫn dị bản
1. **Symptom:** Agent khẳng định có mâu thuẫn trong khi thực tế 2 dị bản bổ sung cho nhau.
2. **Why 1:** Agent cố gắng "tìm kiếm mâu thuẫn" do Prompt yêu cầu quá mạnh.
3. **Why 2:** LLM Judge (GPT-4o) nhận ra đây là lỗi Logic và trừ điểm nặng.
4. **Root Cause:** Prompt Engineering cho Agent quá thiên kiến (Bias) vào việc tìm lỗi sai thay vì trung lập.

## 4. Kế hoạch cải tiến (Action Plan)
- [x] Triển khai Multi-Judge để phát hiện sớm các câu trả lời bị Bias.
- [ ] Áp dụng **Recursive Character Splitting** với overlap lớn hơn để tránh mất context giữa các chunk.
- [ ] Thêm bước **Query Expansion** để làm rõ các đại từ (anh ấy, cô ấy) trước khi tìm kiếm.
- [ ] Tích hợp **Cohere Reranker** để cải thiện Hit Rate từ 40% lên 80%.
