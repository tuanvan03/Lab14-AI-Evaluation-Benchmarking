import asyncio
import time
from typing import List, Dict
# Import other components...

class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()
        
        # 1. Gọi Agent
        chat_history = test_case.get("conversation_history", [])
        response = await self.agent.query(test_case["question"], chat_history=chat_history)
        latency = time.perf_counter() - start_time
        
        # 2. Chạy RAGAS metrics
        ragas_scores = await self.evaluator.score(test_case, response)
        
        # 3. Chạy Multi-Judge
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"], 
            response["answer"], 
            test_case["expected_answer"]
        )
        
        return {
            "test_case": test_case["question"],
            "agent_response": response["answer"],
            "latency": latency,
            "ragas": ragas_scores,
            "judge": judge_result,
            "agent_metadata": response.get("metadata", {}),
            "judge_usage": judge_result.get("usage", {}),
            "status": "fail" if judge_result["final_score"] < 3 else "pass"
        }

    async def run_all(self, dataset: List[Dict], mode: str = "parallel", batch_size: int = 5) -> Dict:
        """
        mode: 'parallel' sử dụng asyncio.gather, 'sequential' chạy từng case một.
        """
        start_time = time.perf_counter()
        results = []
        
        if mode == "parallel":
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
                tasks = [self.run_single_test(case) for case in batch]
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
        else:
            for case in dataset:
                result = await self.run_single_test(case)
                results.append(result)
        
        duration = time.perf_counter() - start_time
        return {
            "results": results,
            "duration": duration,
            "mode": mode
        }
