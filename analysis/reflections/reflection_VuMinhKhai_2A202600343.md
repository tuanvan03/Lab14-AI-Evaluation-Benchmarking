# Báo cáo Cá nhân — Lab 14: AI Evaluation & Benchmarking

**Họ tên:** Vũ Minh Khải
**MSSV:** 2A202600343

**Engineering Contribution** 
1. Cài đặt agent ở main_agent.py với lịch sử chat và hàm rewrite_question trước khi truy vấn tìm tài liệu.
- commit: `b42ff63d7c16a20b34b9c4a61561debbcf3e2446`

2. Cài đặt code ở synthetic_gen.py để sử dụng LLM gpt-4o-mini sinh dữ liệu. Các mẫu trong bộ câu hỏi và câu trả lời chia làm 2 loại single turn `(fact-check, adversarial-injection, edge-conflicting, adversarial-hijack, edge-out-of-context, edge-ambiguous)` và multi-turn `(carry-over, correction)`. Có thể tùy chỉnh số lượng, độ khó mỗi loại. Đầu vào là 1 list các chunk của tài liệu
- Commit: `e71d5cd07f3358e54ae5e1a658b4c5b57deb1ce9`, `dab0262d1f32ddac33f1c71bff4480c0f0ab159c`, `68b82add0afa6df6c53930902cffaca7ef1cdd22`, `9933f3950832e6ab16f8f2115ea1587667ee0e7f`

3. Multi-Judge: (Ko làm phần này nên ko có commit): Góp ý khi thành viên khác sử dụng 2 model gpt-4o và gpt-4o-mini để đánh giá. gpt-4o-mini quá yếu để làm judge — model nhỏ thường không đủ khả năng hiểu sắc thái câu trả lời, dẫn đến chấm điểm thiếu nhất quán và không đáng tin cậy. Tôi góp ý nên thay gpt-4o-mini bằng một model mạnh hơn (ví dụ gpt-4.1, gpt-5) để đảm bảo 2 judge có năng lực tương đương nhau, từ đó kết quả đồng thuận (consensus) mới có ý nghĩa thống kê.

**Technical Depth**
- MRR 
```
MRR: Được sử dụng để đánh giá hiệu suất hệ thống retrieval. MRR đánh giá xem kết quả phù hợp đầu tiên ở vị trí thứ mấy.
- Công thức: MRR = (1/N) × Σ (1/rank_i)
- Với N là số truy vấn, rank_i là số thứ tự vị trí của tài liệu đúng trong các tập các tài liệu truy vấn được ở lần truy vấn thứ i.

Với MRR = 1.0 -> luôn trả đúng kết quả ở vị trí thứ 1: hệ thống retrieval hoajat động rất tốt
Với MRR = 0.5 -> trung bình kết quả trả về ở vị trí thứ 2,...
```

- Cohen's Kappa
```
Cohen's Kappa trả lời câu hỏi: "Hai người đồng ý với nhau — nhưng có phải do họ thực sự đồng thuận, hay chỉ do may mắn?"
- Công thức: k = (P_observed - P_expected) / (1 - P_expected)
- P_observed: sự đồng thuận quan sát được
- P_expected: sự đông thuận do ngẫu nhiên

Với:
- k thuộc 0.4 – 0.6 vừa phải
- k thuộc 0.4 – 0.6 vừa phải
- k thuộc 0.4 – 0.6 vừa phải
```


- Position Bias
```
Là hiện tượng LLM judge có xu hướng chọn câu trả lời ở một vị trí nhất định, bất kể chất lượng thực sự. 
- Ví dụ với 1 câu hỏi có nhiều đáp án, LLM luôn chọn đáp án ở vị trí đầu cho dù có thay đổi nội dung.
Điều này xảy ra vì LLM được train trên văn bản của con người — mà con người có xu hướng đặt thông tin quan trọng ở đầu hoặc cuối. Model học theo pattern đó nên bị bias theo vị trí.

Nếu không kiểm soát position bias, kết quả benchmark của bạn sẽ sai lệch có hệ thống — system nào được đặt ở vị trí đầu sẽ luôn có lợi thế, không liên quan đến chất lượng thực.
Cách xử lý thường dùng: luôn chạy cả 2 chiều rồi lấy kết quả đa số hoặc chỉ tính những cặp mà LLM nhất quán cả 2 chiều.

```

- Trade-off Chi phí vs Chất lượng:
```
Lấy các model Claude làm ví dụ:
- Claude Haiku: chi phí rẻ nhất, tốc độ nhanh, phù hợp cho các tác vụ đơn giản. Tuy nhiên chất lượng không tốt với các câu hỏi phức tạp, suy luận nhiều bước, hoặc làm judge đòi hỏi hiểu sắc thái.
- Claude Sonnet: cân bằng tốt giữa chi phí và chất lượng. Xử lý được các câu hỏi phức tạp, phù hợp làm judge trong pipeline đánh giá RAG. Là lựa chọn thực tế nhất cho hầu hết use case production.
- Claude Opus: chất lượng cao nhất, hiểu sắc thái tốt nhất, phù hợp cho các tác vụ đòi hỏi độ chính xác cao (adversarial cases, edge cases khó). Tuy nhiên chi phí đắt và chậm hơn đáng kể.

Cần câu trả lời chất lượng cao thì chi phí cho model cao, mà nếu chi phí cho model thấp thì chất lượng câu trả lời sẽ thấp.
```


**Problem Solving**
```
Phát hiện bug: Khi agent trả lời các câu hỏi multi-turn, câu trả lời sai khác so với expected answer. Kiểm tra code thì hóa ra code gốc chưa truyền  "conversation_history" đối với các câu multi-turn đân đến agent ko đủ thông tin và trả lời sai. -> bổ sung thêm conversation_history vào nội dung gửi cho LLM đối với các câu multi-turn.
- Commit: 18f2bf1fb9efa8fe5130ed5b9ab6e4c2d68f0f08
```