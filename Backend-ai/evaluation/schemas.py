"""Local, provider-neutral schemas for repeatable CaseLens evaluations."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, JsonValue, model_validator


MetricValue = int | float | bool
EvaluationStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
CaseStatus = Literal["passed", "failed", "error", "skipped"]


class MetricResult(BaseModel):
    """One deterministic or model-assisted score with comparison metadata."""

    name: str = Field(min_length=1)
    value: MetricValue
    passed: bool | None = None
    threshold: MetricValue | None = None
    direction: Literal["higher_is_better", "lower_is_better", "target"] | None = None
    evaluator_version: str | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


class EvaluationConfiguration(BaseModel):
    """Versioned pipeline inputs needed to reproduce and compare a run."""

    pipeline_name: str = Field(min_length=1)
    pipeline_version: str | None = None
    pipeline_configuration: dict[str, JsonValue] = Field(default_factory=dict)
    model_versions: dict[str, str] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    code_version: str | None = None
    environment: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class EvaluationCase(BaseModel):
    """One stable dataset item and its optional expected output."""

    id: str = Field(min_length=1)
    input: dict[str, JsonValue]
    expected_output: JsonValue | None = None
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class EvaluationDataset(BaseModel):
    """A versioned, content-addressable collection of evaluation cases."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)
    description: str | None = None
    cases: list[EvaluationCase] = Field(default_factory=list)
    created_at: datetime | None = None
    parent_version: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def case_ids_must_be_unique(self) -> "EvaluationDataset":
        case_ids = [case.id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("EvaluationDataset case IDs must be unique")
        return self


class EvaluationCaseResult(BaseModel):
    """Output and measurements for one case in an evaluation run."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    case_id: str = Field(min_length=1)
    status: CaseStatus
    output: JsonValue | None = None
    metrics: list[MetricResult] = Field(default_factory=list)
    latency_ms: float | None = Field(default=None, ge=0)
    failure_category: str | None = None
    error: str | None = None
    trace_id: uuid.UUID | None = None
    baseline_case_result_id: uuid.UUID | None = None
    comparison: dict[str, JsonValue] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def metric_names_must_be_unique(self) -> "EvaluationCaseResult":
        names = [metric.name for metric in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("EvaluationCaseResult metric names must be unique")
        return self


class EvaluationRun(BaseModel):
    """One immutable-intent experiment record over a specific dataset version."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    dataset_hash: str = Field(min_length=1)
    configuration: EvaluationConfiguration
    status: EvaluationStatus = "pending"
    case_results: list[EvaluationCaseResult] = Field(default_factory=list)
    metrics: list[MetricResult] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    baseline_run_id: uuid.UUID | None = None
    comparison: dict[str, JsonValue] = Field(default_factory=dict)
    failure_category: str | None = None
    error: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_run_collections_and_time(self) -> "EvaluationRun":
        case_ids = [result.case_id for result in self.case_results]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("EvaluationRun case result IDs must be unique")
        metric_names = [metric.name for metric in self.metrics]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("EvaluationRun metric names must be unique")
        if (
            self.started_at is not None
            and self.finished_at is not None
            and self.finished_at < self.started_at
        ):
            raise ValueError("finished_at must not be earlier than started_at")
        return self
