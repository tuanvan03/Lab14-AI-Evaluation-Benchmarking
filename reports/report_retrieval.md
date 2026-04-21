# Báo cáo Đánh giá Retrieval Stage

## 1. Mục tiêu

Trước khi đánh giá Generation Stage, cần chứng minh **Retrieval Stage** hoạt động đủ tốt — tức là Vector DB truy xuất đúng tài liệu chứa context để LLM trả lời.

Hai chỉ số đánh giá:

| Chỉ số | Định nghĩa |
|--------|------------|
| **Hit Rate** | Tỷ lệ query mà `ground_truth_id` xuất hiện trong **Top-K** kết quả |
| **MRR** (Mean Reciprocal Rank) | Trung bình của `1/rank` — phản ánh mức độ "đúng sớm" |

---

## 2. Cấu hình thử nghiệm

| Tham số | Giá trị |
|---------|---------|
| Vector DB | ChromaDB (`vietnamese_tales` collection) |
| Embedding model | `text-embedding-3-small` (OpenAI) |
| Top-K | 5 |
| Metadata filter | `type = "story"` |
| Số case có Ground Truth | 29 |
| Script đánh giá | `engine/retrieval_eval.py` |

---

## 3. Kết quả tổng thể

| Chỉ số | Giá trị |
|--------|---------|
| **Hit Rate** | **89.66%** (26/29 case) |
| **MRR** | **0.6621** |

> **Nhận xét:** Hit Rate ~90% chứng minh Retrieval Stage hoạt động đủ tốt để tiến hành đánh giá Generation. MRR ~0.66 cho thấy tài liệu đúng thường xuất hiện ở vị trí 1–2 trong Top-5.

---

## 4. Phân tích theo độ khó

| Độ khó | Total | Hit | Hit Rate | MRR |
|--------|------:|----:|--------:|----:|
| medium | 29 | 26 | 89.7% | 0.6621 |

---

## 5. Phân tích 3 Case bị Miss

| # | Câu hỏi | GT ID | Top Retrieved |
|---|---------|-------|--------------|
| 1 | *"Sọ Dừa đã làm gì để chứng minh mình không phải người thường?"* | `sodua.txt_story_2` | story_5, 4, 1, 9, 6 |
| 2 | *"Sọ Dừa đã làm gì để chứng minh bản thân trước phú ông?"* | `sodua.txt_story_2` | story_5, 9, 6, 10, 1 |
| 3 | *"Cô em đã có phản ứng như thế nào với Sọ Dừa?"* | `sodua.txt_story_3` | story_4, 5, 6, 10, 9 |

**Nguyên nhân khả thi:**
- Cả 3 case đều thuộc truyện **Sọ Dừa**, hỏi về các chunk index thấp (story_2, story_3), nhưng embedding kéo về các chunk phần sau (story_5–10).
- Câu hỏi mang tính **hành động mơ hồ** không khớp đủ với text mô tả trong chunk ground truth.

---

## 6. Kết luận

**Retrieval Stage đã hoạt động đủ tốt** (Hit Rate ~90%) để tiến hành đánh giá Generation Stage. Còn dư địa cải thiện ở các câu hỏi hành động mơ hồ liên quan đến các chunk đầu của truyện.
