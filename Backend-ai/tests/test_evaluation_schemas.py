import json
import unittest
import uuid
from datetime import UTC, datetime, timedelta

from pydantic import ValidationError

from evaluation.schemas import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationConfiguration,
    EvaluationDataset,
    EvaluationRun,
    MetricResult,
)


class EvaluationSchemaTests(unittest.TestCase):
    def test_dataset_is_versioned_hashed_and_json_serializable(self):
        dataset = EvaluationDataset(
            name="insurance_retrieval",
            version="v2",
            content_hash="sha256:abc123",
            created_at=datetime(2026, 10, 1, tzinfo=UTC),
            parent_version="v1",
            cases=[EvaluationCase(
                id="Q01",
                input={"query": "Is bicycle theft covered?"},
                expected_output={"expected_chunk_ids": ["chunk-1"]},
                category="bicycle_theft",
            )],
        )

        payload = json.loads(dataset.model_dump_json())
        restored = EvaluationDataset.model_validate(payload)

        self.assertEqual(restored, dataset)
        self.assertEqual(payload["version"], "v2")
        self.assertEqual(payload["content_hash"], "sha256:abc123")

    def test_configuration_records_pipeline_model_and_prompt_versions(self):
        configuration = EvaluationConfiguration(
            pipeline_name="hybrid_bge",
            pipeline_version="v2",
            pipeline_configuration={
                "candidates": 32,
                "batch_size": 8,
                "max_length": 256,
                "fusion_weights": [0.5, 0.5],
            },
            model_versions={
                "embedding": "all-MiniLM-L6-v2",
                "reranker": "BAAI/bge-reranker-v2-m3",
            },
            prompt_versions={"grounded_answer": "v3"},
            code_version="git:abc123",
        )

        self.assertEqual(configuration.pipeline_configuration["candidates"], 32)
        self.assertEqual(configuration.model_versions["reranker"], "BAAI/bge-reranker-v2-m3")
        self.assertEqual(configuration.prompt_versions["grounded_answer"], "v3")

    def test_case_result_supports_metrics_latency_failure_and_trace(self):
        trace_id = uuid.uuid4()
        result = EvaluationCaseResult(
            case_id="Q05",
            status="failed",
            output={"correct_rank": 12},
            metrics=[MetricResult(
                name="hit_at_5",
                value=False,
                passed=False,
                threshold=True,
                direction="target",
            )],
            latency_ms=235.4,
            failure_category="retrieval_miss",
            trace_id=trace_id,
        )

        self.assertEqual(result.latency_ms, 235.4)
        self.assertEqual(result.failure_category, "retrieval_miss")
        self.assertEqual(result.trace_id, trace_id)
        self.assertFalse(result.metrics[0].value)

    def test_run_keeps_dataset_snapshot_and_future_regression_links(self):
        baseline_run_id = uuid.uuid4()
        baseline_case_result_id = uuid.uuid4()
        started = datetime(2026, 10, 1, tzinfo=UTC)
        run = EvaluationRun(
            name="hybrid-bge-candidate",
            dataset_name="insurance_retrieval",
            dataset_version="v2",
            dataset_hash="sha256:abc123",
            configuration=EvaluationConfiguration(pipeline_name="hybrid_bge"),
            status="completed",
            case_results=[EvaluationCaseResult(
                case_id="Q01",
                status="passed",
                baseline_case_result_id=baseline_case_result_id,
                comparison={"rank_delta": 2},
            )],
            metrics=[MetricResult(
                name="mrr",
                value=0.56,
                direction="higher_is_better",
            )],
            started_at=started,
            finished_at=started + timedelta(seconds=2),
            latency_ms=2000,
            baseline_run_id=baseline_run_id,
            comparison={"mrr_delta": 0.06},
        )

        restored = EvaluationRun.model_validate_json(run.model_dump_json())

        self.assertEqual(restored, run)
        self.assertEqual(restored.baseline_run_id, baseline_run_id)
        self.assertEqual(restored.case_results[0].baseline_case_result_id, baseline_case_result_id)
        self.assertEqual(restored.comparison["mrr_delta"], 0.06)

    def test_duplicate_case_and_metric_names_are_rejected(self):
        duplicate_case = EvaluationCase(id="Q01", input={"query": "test"})
        with self.assertRaises(ValidationError):
            EvaluationDataset(
                name="dataset",
                version="v1",
                content_hash="hash",
                cases=[duplicate_case, duplicate_case],
            )

        metric = MetricResult(name="mrr", value=0.5)
        with self.assertRaises(ValidationError):
            EvaluationCaseResult(
                case_id="Q01",
                status="passed",
                metrics=[metric, metric],
            )

    def test_negative_latency_and_invalid_time_range_are_rejected(self):
        with self.assertRaises(ValidationError):
            EvaluationCaseResult(case_id="Q01", status="error", latency_ms=-1)

        now = datetime(2026, 10, 1, tzinfo=UTC)
        with self.assertRaises(ValidationError):
            EvaluationRun(
                name="invalid-run",
                dataset_name="dataset",
                dataset_version="v1",
                dataset_hash="hash",
                configuration=EvaluationConfiguration(pipeline_name="test"),
                started_at=now,
                finished_at=now - timedelta(seconds=1),
            )


if __name__ == "__main__":
    unittest.main()
