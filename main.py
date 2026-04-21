import asyncio
import json
import os
import time
from engine.runner import BenchmarkRunner
from agent.main_agent import MainAgent
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from typing import List, Dict

class ExpertEvaluator:
    def __init__(self):
        self.retrieval_evaluator = RetrievalEvaluator()
        
    async def score(self, case, resp): 
        # Sử dụng RetrievalEvaluator để tính Hit rate & MRR thực tế
        expected_ids = case.get("ground_truth_ids", [])
        retrieved_ids = resp.get("retrieved_ids", [])
        
        hit_rate = self.retrieval_evaluator.calculate_hit_rate(expected_ids, retrieved_ids) if expected_ids else 0.0
        mrr = self.retrieval_evaluator.calculate_mrr(expected_ids, retrieved_ids) if expected_ids else 0.0
        
        # Điểm RAGAS framework giữ nguyên giả lập (có thể kết nối api Ragasonlines sau này)
        return {
            "faithfulness": 0.9, 
            "relevancy": 0.8,
            "retrieval": {"hit_rate": hit_rate, "mrr": mrr}
        }

# Thay thế bằng Judge thực tế mà ta đã làm thay vì dùng class giả lập MultiModelJudge

PRICES = {
    "gpt-4o-mini": {"in": 0.15 / 1_000_000, "out": 0.60 / 1_000_000},
    "gpt-4o": {"in": 10.00 / 1_000_000, "out": 30.00 / 1_000_000},
}

def calculate_costs(batch_results: List[Dict]) -> Dict[str, float]:
    total_cost = 0.0
    total_tokens = 0
    
    for res in batch_results:
        # 1. Agent cost
        meta = res.get("agent_metadata", {})
        model = meta.get("model", "gpt-4o-mini")
        pricing = PRICES.get(model, PRICES["gpt-4o-mini"])
        
        in_tokens = meta.get("prompt_tokens", 0)
        out_tokens = meta.get("completion_tokens", 0)
        
        cost = (in_tokens * pricing["in"]) + (out_tokens * pricing["out"])
        total_cost += cost
        total_tokens += meta.get("tokens_used", 0)
        
        # 2. Judge costs
        judge_usage = res.get("judge_usage", {})
        for j_model, usage in judge_usage.items():
            j_pricing = PRICES.get(j_model, PRICES["gpt-4o-mini"]) # fallback if model name varies
            total_cost += (usage["prompt_tokens"] * j_pricing["in"]) + (usage["completion_tokens"] * j_pricing["out"])
            total_tokens += usage["total_tokens"]

    return {"total_cost": total_cost, "total_tokens": total_tokens}

async def run_benchmark_with_results(agent_version: str, mode: str = "parallel"):
    print(f"[*] Beginning {mode.upper()} Benchmark for {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("[!] Missing data/golden_set.jsonl. Run 'python data/synthetic_gen.py' first.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("[!] File data/golden_set.jsonl is empty.")
        return None, None

    runner = BenchmarkRunner(MainAgent(), ExpertEvaluator(), LLMJudge())
    batch_output = await runner.run_all(dataset, mode=mode)
    results = batch_output["results"]
    duration = batch_output["duration"]

    costs = calculate_costs(results)
    total = len(results)
    
    summary = {
        "metadata": {
            "version": agent_version, 
            "total": total, 
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": round(duration, 2),
            "mode": mode,
            "total_cost_usd": costs["total_cost"],
            "total_tokens": costs["total_tokens"]
        },
        "metrics": {
            "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
            "hit_rate": sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total,
            "avg_mrr": sum(r["ragas"]["retrieval"]["mrr"] for r in results) / total,
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total
        }
    }
    return results, summary

async def main():
    print("=== STARTING COMPREHENSIVE BENCHMARK ===")
    
    # 1. So sánh Sequential vs Parallel cho V1
    print("\n--- PERFORMANCE COMPARISON (V1) ---")
    v1_seq_results, v1_seq_summary = await run_benchmark_with_results("Agent_V1_Base", mode="sequential")
    v1_par_results, v1_par_summary = await run_benchmark_with_results("Agent_V1_Base", mode="parallel")
    
    print(f"\nSequential Time: {v1_seq_summary['metadata']['duration_sec']}s")
    print(f"Parallel Time:   {v1_par_summary['metadata']['duration_sec']}s")
    speedup = v1_seq_summary['metadata']['duration_sec'] / v1_par_summary['metadata']['duration_sec']

    with open("reports/report_compare_sequential_parallel.txt", "w", encoding="utf-8") as f:
        f.write("=== Performance Comparison Report ===\n")
        f.write(f"Version: Agent_V1_Base\n")
        f.write("-" * 40 + "\n")
        
        f.write(f"Sequential Summary:\n{json.dumps(v1_seq_summary, ensure_ascii=False, indent=2)}\n\n")
        f.write(f"Parallel Summary:\n{json.dumps(v1_par_summary, ensure_ascii=False, indent=2)}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write(f"Speedup Factor: {speedup:.2f}x\n")
        f.write("=====================================\n")

    # 2. Chạy V2 Parallel
    print("\n--- OPTIMIZED AGENT (V2) ---")
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized", mode="parallel")
    
    print("\n[=] --- FINAL SUMMARIES ---")
    for s in [v1_par_summary, v2_summary]:
        m = s["metadata"]
        met = s["metrics"]
        print(f"\nVersion: {m['version']} ({m['mode']})")
        print(f"Time: {m['duration_sec']}s | Tokens: {m['total_tokens']} | Cost: ${m['total_cost_usd']:.6f}")
        print(f"Score: {met['avg_score']:.2f} | Hit@3: {met['hit_rate']*100:.1f}% | MRR: {met['avg_mrr']:.2f}")

    # File reports lưu bản V2 Parallel
    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    delta = v2_summary["metrics"]["avg_score"] - v1_par_summary["metrics"]["avg_score"]
    if delta > 0:
        print(f"\n[V] DECISION: APPROVE UPDATE (Delta: +{delta:.2f})")
    else:
        print(f"\n[X] DECISION: REJECT (Delta: {delta:.2f})")

if __name__ == "__main__":
    asyncio.run(main())
