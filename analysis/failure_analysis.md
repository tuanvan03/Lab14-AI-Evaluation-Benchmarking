# Báo cáo Phân tích Thất bại (Failure Analysis)

## 1. Tổng quan Benchmark

| Chỉ số                   | Giá trị        |
| ------------------------ | -------------- |
| Tổng số test case        | 54             |
| Pass                     | **34 (63.0%)** |
| Fail                     | **20 (37.0%)** |
| Avg Judge Score (1–5)    | **2.963**      |
| Avg Agreement Rate       | 0.881          |
| Avg Faithfulness (RAGAS) | 0.900          |
| Avg Relevancy (RAGAS)    | 0.800          |
| Avg Hit Rate (RAGAS)     | 0.407          |
| Avg MRR (RAGAS)          | 0.333          |
| Avg Latency              | 4.16s          |

---

## 2. Phân bổ điểm Judge (1–5)

| Điểm | Số case | Tỷ lệ |
| ---: | :-----: | ----: |
|    1 |   18    | 33.3% |
|    2 |    0    |  0.0% |
|    3 |   15    | 27.8% |
|    4 |    1    |  1.9% |
|    5 |   18    | 33.3% |

> **Nhận xét:** Phân bổ điểm dạng **bi-modal** — tập trung ở hai cực 1 và 5. Không có điểm "trung bình" thực sự, cho thấy Agent hoặc trả lời rất tốt hoặc rất tệ, không có vùng xám.

---

## 3. Phân nhóm lỗi (Failure Clustering)

Dựa trên phân tích 20 case Fail và reasoning của Judge:

| Nhóm lỗi                             | Số case | Nguyên nhân chính                                                                |
| ------------------------------------ | :-----: | -------------------------------------------------------------------------------- |
| **Retrieval Miss — Context sai**     |    8    | Hit Rate = 0, Agent không tìm được đúng chunk → hallucinate hoặc từ chối trả lời |
| **Câu hỏi mơ hồ (Ambiguous)**        |    4    | Agent tự đoán thay vì hỏi lại, dẫn đến trả lời sai đối tượng                     |
| **Edge Conflicting — Phán đoán sai** |    4    | Agent khẳng định sai sự tồn tại của mâu thuẫn trong khi thực tế không có         |
| **Judge mâu thuẫn về Adversarial**   |    4    | gpt-4o-mini chấm điểm khác gpt-4o đối với case hijack/injection (khó phán xét)   |

---

## 4. Phân tích 5-Whys (3 case tệ nhất)

### Case #1: Câu hỏi mơ hồ — "Anh ấy đã lấy gì để lên thiên đình?"

- **Agent trả lời:** *"Anh ấy đã lấy giày làm bằng da bò để lên thiên đình."*
- **Judge reasoning:** *"Câu hỏi không cung cấp thông tin cần thiết để xác định 'anh ấy' là ai."*

| Why            | Nguyên nhân                                                                       |
| -------------- | --------------------------------------------------------------------------------- |
| Why 1          | Agent trả lời thay vì hỏi lại khi Reference mơ hồ ("anh ấy")                      |
| Why 2          | System Prompt chưa đủ mạnh về rule "Nếu mơ hồ → hỏi lại"                          |
| Why 3          | Rewritten query thêm ngữ cảnh sai dựa trên context truy xuất được                 |
| Why 4          | Vector DB kéo về context của truyện Ngưu Lang, không đủ để xác định đúng nhân vật |
| **Root Cause** | **System Prompt + Query Rewriter chưa xử lý tốt câu hỏi thiếu chủ ngữ**           |

---

### Case #2: Edge Conflicting sai — "Hai dị bản về kết quả bò già khỏi bệnh có mâu thuẫn không?"

- **Agent trả lời:** Khẳng định *có* mâu thuẫn (HitRate=1.0, MRR=1.0, chunk đúng).
- **Judge reasoning:** *"Khẳng định có mâu thuẫn trong khi thực tế không có."*

| Why            | Nguyên nhân                                                                                            |
| -------------- | ------------------------------------------------------------------------------------------------------ |
| Why 1          | Agent bịa ra mâu thuẫn khi đọc 2 đoạn văn cùng chủ đề                                                  |
| Why 2          | LLM có xu hướng hallucinate khi bị hỏi về "conflict" giữa 2 đoạn text tương đồng                       |
| Why 3          | System Prompt thiếu instruction "Nếu không có mâu thuẫn rõ ràng → trả lời thẳng là không có"           |
| **Root Cause** | **LLM Hallucination khi trả lời câu hỏi loại edge-conflicting, System Prompt cần rule tường minh hơn** |

---

### Case #3: Retrieval Miss — "Thạch Sanh đã cứu ai từ cũi sắt?"

- **Agent trả lời:** *"Tôi không tìm thấy thông tin trong tài liệu."*
- **Hit Rate = 0, MRR = 0** — Vector DB không truy xuất đúng chunk.

| Why            | Nguyên nhân                                                                                  |
| -------------- | -------------------------------------------------------------------------------------------- |
| Why 1          | Agent từ chối trả lời vì không có context liên quan                                          |
| Why 2          | Hit Rate = 0 → Vector DB không trả về đúng chunk chứa thông tin                              |
| Why 3          | Câu hỏi hỏi về sự kiện cụ thể trong một chunk nhỏ không được index chính xác                 |
| Why 4          | `chunk_size=1000, sentence_overlap=2` tạo ra các chunk quá dài, pha loãng thông tin chi tiết |
| **Root Cause** | **Retrieval Miss do Chunking Strategy chưa tối ưu cho fact-based queries chi tiết**          |

---

## 5. Disagreement giữa Judge Models (14/54 case)

Hai model judge (gpt-4o và gpt-4o-mini) bất đồng ở 14/54 case (**25.9%**). Chủ yếu xảy ra ở:
- **Adversarial cases**: gpt-4o-mini chấm cao hơn khi Agent từ chối lịch sự, gpt-4o chấm thấp hơn vì yêu cầu không được đáp ứng.

---

## 6. Kế hoạch cải tiến (Action Plan)

|    Ưu tiên     | Hành động                                                                                                     | Kỳ vọng                                     |
| :------------: | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
|    **Cao**     | Thêm rule tường minh vào System Prompt: *"Nếu context không có mâu thuẫn rõ ràng, hãy nói thẳng là không có"* | Giảm hallucination ở edge-conflicting       |
|    **Cao**     | Tăng cường rule *"Nếu câu hỏi mơ hồ → hỏi lại"* trong System Prompt + Rewrite Prompt                          | Giảm nhóm lỗi Ambiguous                     |
| **Trung bình** | Giảm `chunk_size` từ 1000 → 500 và tăng `sentence_overlap` → 3 để cải thiện Hit Rate cho chi tiết nhỏ         | Tăng Hit Rate từ 40% → 70%+                 |
| **Trung bình** | Tích hợp **Hybrid Search (Vector + BM25)** để bắt được từ khóa tên riêng và sự kiện cụ thể                    | Cải thiện MRR và Retrieval chất lượng       |
|    **Thấp**    | Thêm **Reranker** sau bước retrieve top-10, chọn top-3 bằng Cross-Encoder                                     | Cải thiện chất lượng context truyền vào LLM |
