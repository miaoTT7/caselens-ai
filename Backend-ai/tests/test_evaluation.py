import unittest

from evaluation.run_evaluation import calculate_metrics, rank_for


class EvaluationMetricTests(unittest.TestCase):
    def test_rank_uses_expected_chunk_ids(self):
        results = [{"chunk_id": "wrong"}, {"chunk_id": "correct"}]
        labeled = {"expected_chunk_ids": ["correct"]}

        self.assertEqual(rank_for(labeled, results), 2)
        self.assertEqual(rank_for({"expected_chunk_ids": ["missing"]}, results), 0)
        self.assertIsNone(rank_for({"expected_chunk_ids": []}, results))

    def test_metrics_calculate_hits_mrr_and_latency(self):
        rows = [
            {"correct_rank": 1, "hit_at_1": True, "hit_at_3": True, "hit_at_5": True, "rr": 1.0, "latency_ms": 10.0},
            {"correct_rank": 3, "hit_at_1": False, "hit_at_3": True, "hit_at_5": True, "rr": 1 / 3, "latency_ms": 20.0},
            {"correct_rank": 0, "hit_at_1": False, "hit_at_3": False, "hit_at_5": False, "rr": 0.0, "latency_ms": 30.0},
        ]

        metrics = calculate_metrics(rows)

        self.assertAlmostEqual(metrics["hit_at_1"], 1 / 3)
        self.assertAlmostEqual(metrics["hit_at_3"], 2 / 3)
        self.assertAlmostEqual(metrics["mrr"], (1 + 1 / 3) / 3)
        self.assertEqual(metrics["average_latency_ms"], 20.0)


if __name__ == "__main__":
    unittest.main()
