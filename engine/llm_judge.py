import asyncio
import os
import json
from typing import Dict, Any
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMJudge:
    def __init__(self, model_a: str = "gpt-4o-mini", model_b: str = "gpt-4o"):
        self.model_a = model_a
        self.model_b = model_b
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        self.rubrics = {
            "accuracy": "Chấm điểm từ 1-5. 5: Chính xác hoàn toàn so với Ground Truth, không sai sót. 3: Đúng một phần nhưng thiếu chi tiết. 1: Sai hoàn toàn hoặc bịa đặt.",
            "tone": "Chấm điểm từ 1-5. 5: Cực kỳ chuyên nghiệp, lịch sự và rõ ràng. 3: Bình thường. 1: Thô lỗ hoặc thiếu tôn trọng."
        }
        
    def _build_prompt(self, question: str, answer: str, ground_truth: str) -> str:
        return f"""Bạn là một giám khảo AI có nhiệm vụ chấm điểm câu trả lời của một AI Agent khác.
Dựa trên câu hỏi, câu trả lời của agent và đáp án chuẩn (Ground Truth), hãy chấm điểm từ 1 đến 5 cho hai tiêu chí sau:
- Accuracy: {self.rubrics["accuracy"]}
- Tone: {self.rubrics["tone"]}

Câu hỏi: {question}
Câu trả lời của Agent: {answer}
Đáp án chuẩn: {ground_truth}

Trích xuất kết quả bằng định dạng JSON bao gồm: "accuracy_score" (int), "tone_score" (int), và "reasoning" (string, giải thích ngắn gọn). Không output thêm bất kỳ text nào khác."""

    async def _call_model(self, model: str, prompt: str) -> Dict[str, Any]:
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=200
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error calling judge {model}: {e}")
            return {"accuracy_score": 3, "tone_score": 3, "reasoning": "Error occurred"}

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        EXPERT TASK: Gọi ít nhất 2 model. Tính toán sự sai lệch. Nếu lệch > 1 điểm, cần logic xử lý.
        """
        prompt = self._build_prompt(question, answer, ground_truth)
        
        # Chạy song song 2 model
        result_a, result_b = await asyncio.gather(
            self._call_model(self.model_a, prompt),
            self._call_model(self.model_b, prompt)
        )
        
        # Tính điểm Accuracy (ta sẽ ưu tiên điểm accuracy là điểm chính)
        score_a = result_a.get("accuracy_score", 3)
        score_b = result_b.get("accuracy_score", 3)
        
        diff = abs(score_a - score_b)
        
        # Thỏa thuận
        if diff == 0:
            agreement = 1.0
            final_score = score_a
            reasoning = f"Cả 2 model đồng thuận. Lý do: {result_a.get('reasoning')}"
        elif diff <= 1:
            agreement = 0.8
            final_score = (score_a + score_b) / 2
            reasoning = f"2 model lệch ít. Điểm trung bình. A lý do: {result_a.get('reasoning')} - B lý do: {result_b.get('reasoning')}"
        else:
            agreement = 0.5
            # Logic xử lý xung đột: lấy điểm thấp hơn để an toàn (Strict Evaluation)
            final_score = min(score_a, score_b)
            reasoning = f"XUNG ĐỘT LỚN. Áp dụng Strict Rule (lấy điểm thấp). A ({score_a}): {result_a.get('reasoning')} - B ({score_b}): {result_b.get('reasoning')}"
            
        return {
            "final_score": final_score,
            "agreement_rate": agreement,
            "individual_scores": {self.model_a: score_a, self.model_b: score_b},
            "reasoning": reasoning
        }

    async def check_position_bias(self, response_a: str, response_b: str):
        """
        Nâng cao: Thực hiện đổi chỗ response A và B để xem Judge có thiên vị vị trí không.
        """
        # Feature này có thể để dạng mở rộng sau
        pass
