import asyncio
import os
from typing import List, Dict
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Mock Vector DB — thay bằng kết nối thật khi có
# ---------------------------------------------------------------------------

MOCK_DOCUMENTS = [
    {
        "id": "doc_001",
        "content": "Để đổi mật khẩu, người dùng truy cập vào trang Cài đặt > Bảo mật > Đổi mật khẩu. Nhập mật khẩu cũ, sau đó nhập mật khẩu mới hai lần để xác nhận. Mật khẩu mới phải có ít nhất 8 ký tự, gồm chữ hoa, chữ thường và số.",
        "source": "policy_handbook.pdf",
    },
    {
        "id": "doc_002",
        "content": "Chính sách hoàn tiền: Khách hàng có thể yêu cầu hoàn tiền trong vòng 30 ngày kể từ ngày mua hàng. Sản phẩm phải còn nguyên vẹn, chưa qua sử dụng. Liên hệ bộ phận CSKH qua email support@company.com hoặc hotline 1800-xxxx.",
        "source": "refund_policy.pdf",
    },
    {
        "id": "doc_003",
        "content": "Tài khoản bị khóa sau 5 lần đăng nhập sai liên tiếp. Để mở khóa, người dùng nhấn 'Quên mật khẩu' trên trang đăng nhập, nhập email đã đăng ký để nhận link đặt lại mật khẩu.",
        "source": "security_guide.pdf",
    },
    {
        "id": "doc_004",
        "content": "Gói dịch vụ Premium bao gồm: hỗ trợ 24/7, lưu trữ không giới hạn, và quyền truy cập tất cả tính năng nâng cao. Giá 199.000 VNĐ/tháng hoặc 1.990.000 VNĐ/năm (tiết kiệm 2 tháng).",
        "source": "pricing.pdf",
    },
    {
        "id": "doc_005",
        "content": "Quy trình onboarding khách hàng mới: (1) Tạo tài khoản, (2) Xác minh email, (3) Hoàn thiện hồ sơ, (4) Chọn gói dịch vụ phù hợp. Đội ngũ tư vấn sẽ liên hệ trong 24 giờ sau khi đăng ký.",
        "source": "onboarding_guide.pdf",
    },
]


def search_vector_db(query: str, top_k: int = 3) -> List[Dict]:
    """
    Mock vector search — trả về top_k tài liệu cố định.
    Thay bằng kết nối ChromaDB/Pinecone/Weaviate khi có vector DB thật.
    """
    # Giả lập relevance score đơn giản dựa trên keyword overlap
    query_words = set(query.lower().split())
    scored = []
    for doc in MOCK_DOCUMENTS:
        doc_words = set(doc["content"].lower().split())
        score = len(query_words & doc_words)
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


# ---------------------------------------------------------------------------
# RAG Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Bạn là trợ lý hỗ trợ khách hàng chuyên nghiệp.
Nhiệm vụ: Trả lời câu hỏi của người dùng DỰA TRÊN context được cung cấp.
Quy tắc:
- Chỉ sử dụng thông tin có trong context. KHÔNG bịa đặt.
- Nếu context không đủ thông tin, hãy nói rõ "Tôi không tìm thấy thông tin về vấn đề này trong tài liệu."
- Trả lời ngắn gọn, rõ ràng, chuyên nghiệp bằng tiếng Việt."""

REWRITE_PROMPT = """Dựa vào lịch sử hội thoại và câu hỏi mới nhất của người dùng, hãy viết lại câu hỏi thành một câu độc lập, đầy đủ ngữ cảnh để dùng truy vấn tìm kiếm tài liệu.
Chỉ trả về câu query đã viết lại, không giải thích thêm."""


class MainAgent:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.name = "SupportAgent-v1"
        self.model = model
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def _rewrite_query(self, question: str, chat_history: List[Dict]) -> str:
        history_text = (
            "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in chat_history)
            if chat_history else "Không có lịch sử."
        )
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": f"Lịch sử:\n{history_text}\n\nCâu hỏi mới: {question}"},
            ],
            temperature=0.0,
            max_tokens=128,
        )
        return response.choices[0].message.content.strip()

    async def query(self, question: str, chat_history: List[Dict] = None) -> Dict:
        """
        chat_history: list of {"role": "user"/"assistant", "content": "..."}
        Dùng cho multi-turn test cases (Context Carry-over, Correction).
        """
        # 1. Query rewriting
        search_query = await self._rewrite_query(question, chat_history or [])

        # 2. Retrieval bằng query đã viết lại
        retrieved_docs = search_vector_db(search_query, top_k=3)
        retrieved_ids = [doc["id"] for doc in retrieved_docs]
        contexts = [doc["content"] for doc in retrieved_docs]
        sources = list({doc["source"] for doc in retrieved_docs})

        context_block = "\n\n".join(
            f"[{doc['id']}] {doc['content']}" for doc in retrieved_docs
        )

        # 2. Build messages với history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if chat_history:
            messages.extend(chat_history)
        messages.append({
            "role": "user",
            "content": f"Context:\n{context_block}\n\nCâu hỏi: {question}",
        })

        # 3. Generation
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=512,
        )

        answer = response.choices[0].message.content
        usage = response.usage

        return {
            "answer": answer,
            "contexts": contexts,
            "retrieved_ids": retrieved_ids,
            "metadata": {
                "model": self.model,
                "tokens_used": usage.total_tokens,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "sources": sources,
                "search_query": search_query,
            },
        }


if __name__ == "__main__":
    async def test():
        agent = MainAgent()

        # Single-turn
        resp = await agent.query("Làm thế nào để đổi mật khẩu?")
        print("Answer:", resp["answer"])
        print("Retrieved IDs:", resp["retrieved_ids"])

        # Multi-turn — Context Carry-over
        history = [
            {"role": "user", "content": "Tôi muốn đổi mật khẩu."},
            {"role": "assistant", "content": resp["answer"]},
        ]
        resp2 = await agent.query("Mật khẩu mới cần đáp ứng điều kiện gì?", chat_history=history)
        print("\nRewritten query:", resp2["metadata"]["search_query"])
        print("Multi-turn answer:", resp2["answer"])

    asyncio.run(test())
