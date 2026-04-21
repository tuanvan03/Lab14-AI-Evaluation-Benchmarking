"""
engine/retrieval_eval.py
========================
Đánh giá Retrieval Stage của hệ thống RAG.
Tính Hit Rate và MRR (Mean Reciprocal Rank) trên bộ dữ liệu golden_set.jsonl.

Cách chạy:  uv run -m engine.retrieval_eval
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

# Cho phép import từ thư mục gốc khi chạy trực tiếp
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.main_agent import search_vector_db


class RetrievalEvaluator:
    """Đánh giá chất lượng Retrieval Stage bằng Hit Rate và MRR."""

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 5) -> float:
        """
        Tính toán xem ít nhất 1 trong expected_ids có nằm trong top_k của retrieved_ids không.
        Trả về 1.0 (HIT) hoặc 0.0 (MISS).
        """
        top_retrieved = retrieved_ids[:top_k]
        return 1.0 if any(doc_id in top_retrieved for doc_id in expected_ids) else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Tính Reciprocal Rank của 1 query.
        Tìm vị trí đầu tiên (1-indexed) của expected_id trong retrieved_ids.
        MRR = 1 / rank. Nếu không tìm thấy thì trả về 0.
        """
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in expected_ids:
                return 1.0 / rank
        return 0.0

    def evaluate_batch(self, dataset: List[Dict], top_k: int = 5, query_type: str = "story") -> Dict:
        """
        Chạy eval cho toàn bộ bộ dữ liệu.
        Chỉ đánh giá các case có ground_truth_ids.

        Args:
            dataset: list các test case đã load từ golden_set.jsonl
            top_k: số lượng kết quả retrieve từ VectorDB
            query_type: filter metadata theo type ("story" hoặc "lesson")

        Returns:
            Dict chứa avg_hit_rate, avg_mrr, và danh sách kết quả chi tiết
        """
        # Lọc ra những case có ground truth để đánh giá
        evaluable = [d for d in dataset if d.get("ground_truth_ids")]
        if not evaluable:
            print("Cảnh báo: Không có case nào có ground_truth_ids để đánh giá!")
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0, "details": []}

        total = len(evaluable)
        hit_sum = 0.0
        mrr_sum = 0.0
        detail_results = []

        for idx, case in enumerate(evaluable):
            question = case["question"]
            expected_ids = case["ground_truth_ids"]

            # Truy xuất thực tế từ Vector DB
            retrieved = search_vector_db(query=question, top_k=top_k, type=query_type)
            retrieved_ids = [r["id"] for r in retrieved]

            # Tính Hit Rate và Reciprocal Rank cho case này
            hit = self.calculate_hit_rate(expected_ids, retrieved_ids, top_k=top_k)
            rr = self.calculate_mrr(expected_ids, retrieved_ids)
            hit_sum += hit
            mrr_sum += rr

            detail_results.append({
                "question": question,
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids,
                "hit": bool(hit),
                "reciprocal_rank": rr,
                "difficulty": case.get("metadata", {}).get("difficulty", "unknown"),
                "type": case.get("metadata", {}).get("type", "unknown"),
            })

            # In tiến độ
            if (idx + 1) % 5 == 0:
                print(f"  ... đã xử lý {idx + 1}/{total} cases")

        return {
            "avg_hit_rate": hit_sum / total,
            "avg_mrr": mrr_sum / total,
            "total_evaluated": total,
            "details": detail_results,
        }


def print_report(results: Dict, top_k: int):
    """In báo cáo chi tiết ra màn hình."""
    hit_rate = results["avg_hit_rate"]
    mrr = results["avg_mrr"]
    details = results.get("details", [])
    total = results.get("total_evaluated", 0)
    hits = sum(1 for d in details if d["hit"])

    print("\n" + "=" * 50)
    print("    KẾT QUẢ ĐÁNH GIÁ RETRIEVAL STAGE")
    print("=" * 50)
    print(f"  Tổng số case (có ground truth): {total}")
    print(f"  Top-K retrieve:                 {top_k}")
    print(f"  Số case HIT:                    {hits}")
    print("-" * 50)
    print(f"  Hit Rate:  {hit_rate:.4f}  ({hit_rate*100:.2f}%)")
    print(f"  MRR:       {mrr:.4f}")
    print("=" * 50)

    # Phân tích theo độ khó
    diff_stats = defaultdict(lambda: {"total": 0, "hit": 0, "mrr_sum": 0.0})
    for r in details:
        d = r["difficulty"]
        diff_stats[d]["total"] += 1
        if r["hit"]:
            diff_stats[d]["hit"] += 1
        diff_stats[d]["mrr_sum"] += r["reciprocal_rank"]

    print("\n--- Phân tích theo Độ khó ---")
    print(f"  {'Độ khó':<12} {'Total':>6} {'Hit':>6} {'HitRate%':>10} {'MRR':>8}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*10} {'-'*8}")
    for diff in ["easy", "medium", "hard"]:
        s = diff_stats.get(diff)
        if not s:
            continue
        hr = s["hit"] / s["total"] if s["total"] > 0 else 0.0
        m = s["mrr_sum"] / s["total"] if s["total"] > 0 else 0.0
        print(f"  {diff:<12} {s['total']:>6} {s['hit']:>6} {hr*100:>9.1f}% {m:>8.4f}")

    # Các case bị MISS
    miss_cases = [r for r in details if not r["hit"]]
    if miss_cases:
        print(f"\n--- {len(miss_cases)} Case bị Miss (hiện 5 case đầu) ---")
        for c in miss_cases[:5]:
            print(f"  Q: {c['question'][:80]}...")
            print(f"     GT IDs   : {c['expected_ids']}")
            print(f"     Retrieved: {c['retrieved_ids']}")
    print()


def main():
    # Đường dẫn đến golden set (relative to project root)
    golden_path = Path(__file__).resolve().parent.parent / "data" / "golden_set.jsonl"
    if not golden_path.exists():
        print(f"Lỗi: Không tìm thấy file {golden_path}")
        return

    with open(golden_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    evaluable = [d for d in dataset if d.get("ground_truth_ids")]
    print(f"Đã tải {len(dataset)} test cases từ golden_set.jsonl")
    print(f"  - Có {len(evaluable)} case có ground_truth_ids để đánh giá retrieval")
    print(f"  - Có {len(dataset) - len(evaluable)} case không có ground truth (adversarial/edge)\n")

    TOP_K = 5
    evaluator = RetrievalEvaluator()

    print(f"Bắt đầu đánh giá với top_k={TOP_K}...")
    results = evaluator.evaluate_batch(dataset, top_k=TOP_K, query_type="story")

    print_report(results, top_k=TOP_K)

    # Lưu kết quả chi tiết ra file JSON
    output_path = Path(__file__).resolve().parent.parent / "data" / "retrieval_eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "hit_rate": results["avg_hit_rate"],
            "mrr": results["avg_mrr"],
            "top_k": TOP_K,
            "total_evaluated": results["total_evaluated"],
            "details": results["details"],
        }, f, ensure_ascii=False, indent=2)
    print(f"Kết quả chi tiết đã lưu vào: {output_path}")


if __name__ == "__main__":
    main()
