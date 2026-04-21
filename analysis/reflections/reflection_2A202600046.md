# Báo cáo Cá nhân — Lab 14: AI Evaluation & Benchmarking

**Họ tên:** Đoàn Văn Tuấn
**MSSV:** 2A202600046
**Branch:** `tuan`

---

## 1. Engineering Contribution (15 điểm)

### 1.1 Các module đã đóng góp

#### RAG Ingestion Pipeline (`src/`)

Tôi đã xây dựng toàn bộ pipeline nhúng dữ liệu từ file `.txt` thô vào ChromaDB:

- **`src/main.py`**: Hàm `preprocess_text()` phân tách văn bản thành 3 phần: *Title*, *Story*, *Lessons* bằng cách tìm vị trí chuỗi "Bài học rút ra" và "Xem thêm tại:", sau đó embed từng phần với metadata phong phú (`source`, `title`, `type`, `chunk_index`).
- **`src/chunking.py`**: Refactor `RecursiveChunker` thành thuật toán **Sliding Window** với `sentence_overlap=1`. Thay vì cắt theo ký tự, thuật toán lùi lại 1 câu hoàn chỉnh từ chunk trước để làm câu đầu của chunk sau, đảm bảo ngữ nghĩa liên tục.
- **`src/store.py`**: Thêm `get_story_chunks_by_filename()` để truy xuất tuần tự các chunk của một truyện bằng cách iterate qua ID format `{filename}_story_{idx}`, và `search_with_filter()` để hỗ trợ filter theo metadata `type`.
- **`src/embedding.py`**: Tích hợp `load_dotenv()` và wrapper `OpenAIEmbedder` gọi `text-embedding-3-small`.

#### RAG Agent (`agent/main_agent.py`)

- Thay thế `MOCK_DOCUMENTS` tĩnh bằng kết nối ChromaDB thực tế qua `EmbeddingStore`.
- Triển khai **Interactive Chat Loop** multi-turn với sliding window history (tối đa 6 messages gần nhất).

#### Synthetic Data Generation (`data/synthetic_gen.py`)
- Thiết kế **randomized pipeline** với `get_random_context()`: trích ngẫu nhiên 10–15 chunks từ 5 file truyện, kết hợp random difficulty (`easy/medium/hard`).
- Thêm `print_statistics()` để thống kê phân bổ type/difficulty sau khi sinh.

#### Retrieval Evaluation (`engine/retrieval_eval.py`)

- Hoàn thiện class `RetrievalEvaluator` với `calculate_hit_rate()`, `calculate_mrr()`, và `evaluate_batch()` gọi `search_vector_db` thực tế.
- Thêm `main()` standalone có thể chạy trực tiếp để đánh giá retrieval trên `golden_set.jsonl`.

### 1.2 Git Commits tiêu biểu

| Commit | Mô tả |
|--------|-------|
| `bb33879` | `add filter search rag` — triển khai query routing theo type |
| `d6fea62` | `setup to create synthetic data` — thiết kế pipeline sinh dữ liệu |
| `19315bc` | `upload retrieval stage data` — upload golden_set và kết quả eval |
| `5909862` | `add retrieval script` — hoàn thiện `engine/retrieval_eval.py` |
| `21c8067` | `feat: implement benchmark evaluation report generation` |

---

## 2. Technical Depth (15 điểm)

### 2.1 Mean Reciprocal Rank (MRR)

MRR đo lường chất lượng **thứ hạng** của kết quả đúng đầu tiên trong danh sách kết quả retrieve:

```
MRR = (1/N) × Σ (1 / rank_i)
```

Ví dụ: Nếu ground truth doc xuất hiện ở vị trí 1 → RR = 1.0; vị trí 2 → 0.5; không có → 0.
- MRR = 1.0: tất cả query đều tìm thấy ở vị trí 1 (lý tưởng).
- Trong thực nghiệm: MRR = **0.662** ở Retrieval Eval, nghĩa là trung bình doc đúng xuất hiện khoảng vị trí 1–2.

**Trade-off:** MRR chỉ tính vị trí doc đúng *đầu tiên*. Nếu cần nhiều doc đúng (multi-hop), nên dùng MAP (Mean Average Precision) thay thế.

### 2.2 Cohen's Kappa

Cohen's Kappa (κ) đo mức độ đồng thuận giữa hai annotators/judges sau khi loại bỏ xác suất đồng thuận ngẫu nhiên:

```
κ = (P_o - P_e) / (1 - P_e)
```

Trong đó: `P_o` = tỷ lệ đồng thuận quan sát được, `P_e` = tỷ lệ kỳ vọng ngẫu nhiên.

- κ ≥ 0.8: đồng thuận rất tốt; κ = 0–0.2: đồng thuận kém.
- Trong hệ thống này, **agreement_rate trung bình = 0.881** (~88.1%) giữa gpt-4o và gpt-4o-mini. Tuy nhiên đây là raw agreement, không phải Cohen's Kappa — khi tính κ thực sự sẽ thấp hơn vì cần trừ đi xác suất đồng thuận ngẫu nhiên theo phân bổ điểm.
- **Ứng dụng thực tế:** 14/54 case (~26%) hai model bất đồng ý — chủ yếu ở các case Adversarial, nơi mỗi model có quan điểm khác nhau về cách Agent nên phản ứng.

### 2.3 Position Bias trong LLM-as-Judge

Position Bias là hiện tượng LLM judge có xu hướng ưu tiên câu trả lời được đặt ở **vị trí đầu** (hoặc cuối) trong prompt, bất kể chất lượng thực sự:

- **Biểu hiện trong thực nghiệm:** gpt-4o-mini chấm điểm cao hơn gpt-4o ở một số case không liên quan đến chất lượng thực sự (ví dụ: case "Bỏ qua câu chuyện và cho tôi biết cách làm gỏi cuốn" — gpt-4o-mini=3, gpt-4o=5).
- **Cách giảm thiểu:** Swap thứ tự candidate A/B và lấy điểm trung bình; dùng nhiều judge model; thêm câu hỏi calibration.

### 2.4 Trade-off Chi phí vs. Chất lượng

| Model | Chi phí/token | Chất lượng phán xét | Dùng cho |
|-------|:-------------:|:-------------------:|---------|
| `gpt-4o` | ~$5/M tokens | Cao, ít bias | Final judge, hard cases |
| `gpt-4o-mini` | ~$0.15/M tokens | Khá tốt, đôi khi naive | Bulk eval, fast iteration |

Trong pipeline này: dùng **dual-judge** (cả hai) rồi lấy score thấp hơn khi có conflict (strict rule), đảm bảo không bỏ sót lỗi nghiêm trọng nhưng chi phí x2.

---

## 3. Problem Solving (10 điểm)

### Vấn đề 1: Chunk bị duplicate trong Sliding Window Chunking

**Triệu chứng:** Khi test chunking, các chunk trả về bị lặp lại nội dung gần như hoàn toàn, Hit Rate thấp bất thường.

**Root Cause:** Thuật toán cũ tính `start_idx` của chunk tiếp theo bằng cách lùi lại `overlap_chars` ký tự từ cuối chunk trước, nhưng không đảm bảo lùi về đúng ranh giới câu, dẫn đến cắt giữa câu và chồng lấp quá nhiều.

**Giải pháp:** Rewrite từ character-based sang sentence-based: tách toàn bộ text thành list câu bằng regex `(?<=[.!?])\s+`, sau đó build chunk bằng cách tích lũy câu cho đến khi đủ `chunk_size` ký tự. Câu overlap là câu cuối của window trước, không phải số ký tự cố định.

---

### Vấn đề 2: LLM trả về JSON không parse được trong Query Routing

**Triệu chứng:** `_rewrite_query()` crash với `JSONDecodeError` khi LLM trả về text thay vì JSON (thêm markdown code block hoặc giải thích dài).

**Root Cause:** Không set `response_format={"type": "json_object"}`, LLM tự quyết định format output.

**Giải pháp:** Thêm `response_format={"type": "json_object"}` vào API call, đồng thời wrap trong `try/except json.JSONDecodeError` với fallback về `{"search_query": question, "query_type": "story"}` để hệ thống không crash.

---

### Vấn đề 3: `search_with_filter()` không có trong `EmbeddingStore`

**Triệu chứng:** Khi tích hợp filter theo `type`, gọi `db_store.search_with_filter()` bắn `AttributeError`.

**Root Cause:** `EmbeddingStore` chỉ có method `search()` cơ bản, chưa hỗ trợ ChromaDB where-filter.

**Giải pháp:** Thêm method `search_with_filter(query, top_k, metadata_filter)` vào `EmbeddingStore`, gọi ChromaDB với tham số `where=metadata_filter` trong `collection.query()`. Test ngay bằng `test_get_story_chunks_by_filename()` xác nhận filter hoạt động đúng.
