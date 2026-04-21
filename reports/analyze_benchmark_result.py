"""
reports/analyze_benchmark_result.py
=====================================
Script phân tích kết quả Benchmark từ file reports/benchmark_results.json.

Phân tích các chỉ số:
  - Tổng quan: pass/fail rate, avg latency, avg judge score
  - Phân tích theo status (pass/fail)
  - Phân tích theo từng Judge Model (gpt-4o-mini vs gpt-4o)
  - Phân bổ điểm Judge (histogram)
  - Top case tốt nhất / tệ nhất
  - Phân tích disagreement giữa các models

Cách chạy: python3 reports/analyze_benchmark_result.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

# ────────────────────────────────────────────────
# Load data
# ────────────────────────────────────────────────

def load_results(path: str) -> list[dict]:
    """Tải kết quả benchmark từ file JSON."""
    p = Path(path)
    if not p.exists():
        print(f"Lỗi: Không tìm thấy file '{path}'")
        sys.exit(1)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("Lỗi: File JSON phải là một array!")
        sys.exit(1)
    return data


# ────────────────────────────────────────────────
# Các hàm phân tích
# ────────────────────────────────────────────────

def overview(results: list[dict]) -> None:
    """In tổng quan kết quả."""
    total = len(results)
    if total == 0:
        print("Không có kết quả nào!")
        return

    passes = sum(1 for r in results if r.get("status") == "pass")
    fails  = total - passes
    avg_latency     = sum(r.get("latency", 0) for r in results) / total
    avg_judge_score = sum(r.get("judge", {}).get("final_score", 0) for r in results) / total
    avg_faithfulness = sum(r.get("ragas", {}).get("faithfulness", 0) for r in results) / total
    avg_relevancy    = sum(r.get("ragas", {}).get("relevancy", 0) for r in results) / total
    avg_hit_rate     = sum(r.get("ragas", {}).get("retrieval", {}).get("hit_rate", 0) for r in results) / total
    avg_mrr          = sum(r.get("ragas", {}).get("retrieval", {}).get("mrr", 0) for r in results) / total
    avg_agreement    = sum(r.get("judge", {}).get("agreement_rate", 0) for r in results) / total

    print("=" * 55)
    print("          TỔNG QUAN BENCHMARK")
    print("=" * 55)
    print(f"  Tổng số test case          : {total}")
    print(f"  Pass                       : {passes}  ({passes/total*100:.1f}%)")
    print(f"  Fail                       : {fails}  ({fails/total*100:.1f}%)")
    print("-" * 55)
    print(f"  Avg Latency (s)            : {avg_latency:.3f}")
    print("-" * 55)
    print(f"  Avg Judge Score (1-5)      : {avg_judge_score:.3f}")
    print(f"  Avg Agreement Rate         : {avg_agreement:.3f}")
    print("-" * 55)
    print(f"  Avg Faithfulness (RAGAS)   : {avg_faithfulness:.3f}")
    print(f"  Avg Relevancy    (RAGAS)   : {avg_relevancy:.3f}")
    print(f"  Avg Hit Rate     (RAGAS)   : {avg_hit_rate:.3f}")
    print(f"  Avg MRR          (RAGAS)   : {avg_mrr:.3f}")
    print("=" * 55)


def score_distribution(results: list[dict]) -> None:
    """In phân bổ điểm Judge (1-5)."""
    counter = Counter(r.get("judge", {}).get("final_score", 0) for r in results)
    total = len(results)
    print("\n--- Phân bổ điểm Judge (1-5) ---")
    for score in range(1, 6):
        cnt = counter.get(score, 0)
        bar = "█" * cnt
        print(f"  [{score}] {bar:<30} {cnt:>4} ({cnt/total*100:5.1f}%)")


def model_comparison(results: list[dict]) -> None:
    """So sánh điểm giữa các judge models."""
    model_totals: dict[str, list] = defaultdict(list)
    for r in results:
        individual = r.get("judge", {}).get("individual_scores", {})
        for model, score in individual.items():
            model_totals[model].append(score)

    if not model_totals:
        return

    print("\n--- So sánh Judge Model ---")
    print(f"  {'Model':<20} {'Avg Score':>10} {'Min':>5} {'Max':>5} {'Count':>7}")
    print(f"  {'-'*20} {'-'*10} {'-'*5} {'-'*5} {'-'*7}")
    for model, scores in sorted(model_totals.items()):
        print(f"  {model:<20} {sum(scores)/len(scores):>10.3f} {min(scores):>5} {max(scores):>5} {len(scores):>7}")


def disagreement_analysis(results: list[dict]) -> None:
    """Phân tích các case mà các model judge không đồng ý."""
    disagreements = []
    for r in results:
        individual = r.get("judge", {}).get("individual_scores", {})
        scores = list(individual.values())
        if len(scores) >= 2 and max(scores) != min(scores):
            disagreements.append({
                "question": r.get("test_case", "")[:70],
                "scores": individual,
                "final_score": r.get("judge", {}).get("final_score"),
                "agreement_rate": r.get("judge", {}).get("agreement_rate"),
            })

    total = len(results)
    print(f"\n--- Disagreement Analysis ({len(disagreements)}/{total} cases) ---")
    if not disagreements:
        print("  Tất cả case đều đồng thuận!")
        return
    for d in disagreements[:10]:
        scores_str = ", ".join(f"{m}={s}" for m, s in d["scores"].items())
        print(f"  Q: {d['question']}...")
        print(f"     Scores: {scores_str}  →  Final: {d['final_score']}  (Agreement: {d['agreement_rate']:.0%})")


def best_worst_cases(results: list[dict], top_n: int = 5) -> None:
    """In top N case tốt nhất và tệ nhất theo Judge Score."""
    sorted_r = sorted(results, key=lambda r: r.get("judge", {}).get("final_score", 0))

    print(f"\n--- {top_n} Case TỆ NHẤT ---")
    for r in sorted_r[:top_n]:
        score = r.get("judge", {}).get("final_score", "?")
        latency = r.get("latency", 0)
        q = r.get("test_case", "")
        print(f"  [Score={score}] [{latency:.2f}s] {q}")

    print(f"\n--- {top_n} Case TỐT NHẤT ---")
    for r in sorted_r[-top_n:][::-1]:
        score = r.get("judge", {}).get("final_score", "?")
        latency = r.get("latency", 0)
        q = r.get("test_case", "")
        print(f"  [Score={score}] [{latency:.2f}s] {q}")


def latency_analysis(results: list[dict]) -> None:
    """Phân tích độ trễ (latency)."""
    latencies = [r.get("latency", 0) for r in results]
    if not latencies:
        return
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    p50 = latencies_sorted[n // 2]
    p90 = latencies_sorted[int(n * 0.9)]
    p95 = latencies_sorted[int(n * 0.95)]
    print(f"\n--- Phân tích Latency ---")
    print(f"  Min   : {min(latencies):.3f}s")
    print(f"  Max   : {max(latencies):.3f}s")
    print(f"  Avg   : {sum(latencies)/n:.3f}s")
    print(f"  P50   : {p50:.3f}s")
    print(f"  P90   : {p90:.3f}s")
    print(f"  P95   : {p95:.3f}s")


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────

def main():
    result_path = Path(__file__).parent / "benchmark_results.json"
    results = load_results(str(result_path))
    print(f"\nĐã tải {len(results)} kết quả từ: {result_path}\n")

    overview(results)
    score_distribution(results)
    model_comparison(results)
    disagreement_analysis(results)
    best_worst_cases(results, top_n=5)
    latency_analysis(results)
    print()


if __name__ == "__main__":
    main()
