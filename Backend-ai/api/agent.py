"""Minimal deterministic agent state and routing over the existing claim workflow."""

from __future__ import annotations

import uuid
from decimal import Decimal
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, Field

from api.claim_schemas import (
    ApplicablePolicyAssessment,
    CalculationResult,
    Claim,
    ClaimAssessmentRequest,
    ClaimAssessmentResponse,
    ClaimCalculationRequest,
    ClaimFact,
    ClaimRecommendationRequest,
    ClaimRecommendation,
    CoverageAssessment,
    CoverageAssessmentRequest,
    ExclusionAssessment,
    ExclusionAssessmentRequest,
    EvidenceRef,
    MissingInformation,
    ObligationAssessment,
    ObligationAssessmentRequest,
    FactValidationRequest,
)


PhaseName = Literal[
    "claim_facts_extraction",
    "fact_validation",
    "coverage_assessment",
    "exclusion_assessment",
    "obligation_assessment",
    "claim_calculation",
    "claim_recommendation",
]

NextAction = Literal[
    "continue_assessment",
    "ask_for_information",
    "rerun_phase",
    "human_review",
    "completed",
]

AnswerValue = str | bool | int | Decimal


class MissingInformationAnswer(BaseModel):
    missing_information_id: uuid.UUID
    value: AnswerValue
    original_text: str | None = None


class MissingInformationAnswerRequest(BaseModel):
    answers: list[MissingInformationAnswer] = Field(min_length=1)


class CaseLensAgentState(BaseModel):
    knowledge_base_id: uuid.UUID | None = None
    retrieval_limit: int = Field(default=8, ge=1, le=20)
    claimant_reference_required: bool = False
    policy_reference_required: bool = False
    claim: Claim | None = None
    facts: list[ClaimFact] = Field(default_factory=list)
    missing_information: list[MissingInformation] = Field(default_factory=list)
    applicable_policy: ApplicablePolicyAssessment | None = None
    coverage_assessments: list[CoverageAssessment] = Field(default_factory=list)
    exclusion_assessments: list[ExclusionAssessment] = Field(default_factory=list)
    obligation_assessments: list[ObligationAssessment] = Field(default_factory=list)
    calculation_results: list[CalculationResult] = Field(default_factory=list)
    recommendation: ClaimRecommendation | None = None
    completed_phases: list[PhaseName] = Field(default_factory=list)
    provider_errors: list[str] = Field(default_factory=list)
    rerun_failure_count: int = Field(default=0, ge=0)
    current_phase: PhaseName | None = None
    next_action: NextAction = "continue_assessment"

    @classmethod
    def from_assessment(cls, response: ClaimAssessmentResponse) -> "CaseLensAgentState":
        current_phase = response.failed_phase or (
            response.completed_phases[-1] if response.completed_phases else None
        )
        requested_action: NextAction = (
            "rerun_phase" if response.error and response.failed_phase else "continue_assessment"
        )
        return cls(
            claim=response.claim,
            facts=response.facts,
            missing_information=response.missing_information,
            applicable_policy=response.applicable_policy,
            coverage_assessments=response.coverage_assessments,
            exclusion_assessments=response.exclusion_assessments,
            obligation_assessments=response.obligation_assessments,
            calculation_results=response.calculation_results,
            recommendation=response.recommendation,
            completed_phases=response.completed_phases,
            provider_errors=(
                [response.error]
                if response.error and "LLMProviderError" in response.error
                else []
            ),
            current_phase=current_phase,
            next_action=requested_action,
        )


class CaseLensAgentRouter:
    @staticmethod
    def route(state: CaseLensAgentState) -> CaseLensAgentState:
        recommendation = state.recommendation
        if recommendation is not None and (
            recommendation.human_review_required
            or recommendation.status == "needs_human_review"
        ):
            action: NextAction = "human_review"
        elif any(
            item.metadata.get("assessment_unavailable") is True
            for item in state.missing_information
        ):
            action = "human_review"
        elif any(item.blocking for item in state.missing_information):
            action = "ask_for_information"
        elif state.next_action == "rerun_phase":
            action = "rerun_phase"
        elif recommendation is not None:
            action = "completed"
        else:
            action = "continue_assessment"
        return state.model_copy(update={"next_action": action})


class MissingInformationInteraction:
    _PHASE_BY_REQUIREMENT: dict[str, PhaseName] = {
        "fact_validation": "fact_validation",
        "policy_applicability": "coverage_assessment",
        "coverage_assessment": "coverage_assessment",
        "exclusion_assessment": "exclusion_assessment",
        "obligation_assessment": "obligation_assessment",
        "claim_calculation": "claim_calculation",
    }
    _PHASE_ORDER = {
        phase: index
        for index, phase in enumerate((
            "fact_validation",
            "coverage_assessment",
            "exclusion_assessment",
            "obligation_assessment",
            "claim_calculation",
        ))
    }

    @staticmethod
    def questions(state: CaseLensAgentState) -> list[MissingInformation]:
        if state.next_action != "ask_for_information":
            return []
        return [item for item in state.missing_information if item.blocking]

    @classmethod
    def apply_answers(
        cls,
        state: CaseLensAgentState,
        request: MissingInformationAnswerRequest,
    ) -> CaseLensAgentState:
        if state.next_action != "ask_for_information":
            raise ValueError("The agent is not waiting for missing-information answers")

        missing_by_id = {
            item.id: item for item in state.missing_information if item.blocking
        }
        answer_ids = [item.missing_information_id for item in request.answers]
        if len(answer_ids) != len(set(answer_ids)):
            raise ValueError("Each missing-information item may be answered only once")

        resolved: list[tuple[MissingInformationAnswer, MissingInformation, PhaseName]] = []
        for answer in request.answers:
            missing = missing_by_id.get(answer.missing_information_id)
            if missing is None:
                raise ValueError(
                    f"Unknown or non-blocking missing-information ID: {answer.missing_information_id}"
                )
            target = cls._PHASE_BY_REQUIREMENT.get(missing.required_for)
            if target is None:
                raise ValueError(
                    f"Unsupported missing-information requirement: {missing.required_for}"
                )
            resolved.append((answer, missing, target))

        facts_by_path = {fact.fact_path: fact for fact in state.facts}
        for answer, missing, _ in resolved:
            evidence_id = f"[C-followup-{uuid.uuid4()}]"
            quote = answer.original_text if answer.original_text is not None else str(answer.value)
            facts_by_path[missing.field_path] = ClaimFact(
                id=uuid.uuid4(),
                fact_path=missing.field_path,
                value=answer.value,
                status="confirmed",
                confidence=None,
                claim_evidence=[EvidenceRef(
                    id=evidence_id,
                    evidence_type="claim",
                    source_file="user_follow_up",
                    text_quote=quote,
                    metadata={"missing_information_id": str(missing.id)},
                )],
                metadata={
                    "missing_information_id": str(missing.id),
                    "related_incident_id": (
                        str(missing.related_incident_id) if missing.related_incident_id else None
                    ),
                    "related_exposure_id": (
                        str(missing.related_exposure_id) if missing.related_exposure_id else None
                    ),
                },
            )

        answered_ids = set(answer_ids)
        target_phase = min(
            (target for _, _, target in resolved),
            key=cls._PHASE_ORDER.__getitem__,
        )
        return state.model_copy(update={
            "facts": list(facts_by_path.values()),
            "missing_information": [
                item for item in state.missing_information if item.id not in answered_ids
            ],
            "current_phase": target_phase,
            "next_action": "rerun_phase",
        })


class CaseLensAgent:
    """Treat the existing Phase 1–7 orchestrator as one reusable assessment tool."""

    def __init__(
        self,
        orchestrator,
        *,
        router: CaseLensAgentRouter | None = None,
        tool_registry=None,
        handoff_builder=None,
        trace_collector=None,
        history_service=None,
        assessment_version_service=None,
    ):
        self.orchestrator = orchestrator
        self.router = router or CaseLensAgentRouter()
        self.tool_registry = tool_registry
        self.handoff_builder = handoff_builder
        self.history_service = history_service
        self.assessment_version_service = assessment_version_service
        if trace_collector is None:
            from api.agent_tracing import TraceCollector

            trace_collector = TraceCollector()
        self.trace_collector = trace_collector

    @property
    def trace(self):
        try:
            return self.trace_collector.snapshot()
        except Exception:
            return None

    async def assess(self, request: ClaimAssessmentRequest) -> CaseLensAgentState:
        try:
            self.trace_collector.start_trace()
        except Exception:
            pass
        self._trace_emit("agent_started", metadata={"knowledge_base_id": str(request.knowledge_base_id)})
        response = await self.orchestrator.assess(request)
        state = CaseLensAgentState.from_assessment(response).model_copy(update={
            "knowledge_base_id": request.knowledge_base_id,
            "retrieval_limit": request.retrieval_limit,
            "claimant_reference_required": request.claimant_reference_required,
            "policy_reference_required": request.policy_reference_required,
        })
        state = self.router.route(state)
        initial_events = []
        if self.history_service is not None:
            initial_events = await self.history_service.record_initial_assessment(
                fnol_text=request.fnol_text,
                state=state,
                assessment_session_id=self._history_session_id(),
            )
        if (
            self.assessment_version_service is not None
            and self.assessment_version_service.is_stable_assessment(state)
        ):
            trigger_reference_id = next(
                (
                    event.event_id
                    for event in initial_events
                    if event.event_type == "fnol_submitted"
                ),
                None,
            )
            await self.assessment_version_service.create_version(
                state,
                trigger_type="initial_assessment",
                trigger_reference_id=trigger_reference_id,
            )
        self._trace_routing(state)
        self._trace_terminal_state(state)
        return state

    async def apply_missing_information_answers(
        self,
        state: CaseLensAgentState,
        request: MissingInformationAnswerRequest,
    ) -> CaseLensAgentState:
        updated = MissingInformationInteraction.apply_answers(state, request)
        if self.history_service is not None:
            await self.history_service.record_answers(
                before=state,
                after=updated,
                request=request,
                assessment_session_id=self._history_session_id(),
            )
        self._trace_emit(
            "user_answer_applied",
            current_phase=updated.current_phase,
            next_action=updated.next_action,
            success=True,
            metadata={
                "answer_count": len(request.answers),
                "missing_information_ids": [
                    str(item.missing_information_id) for item in request.answers
                ],
            },
        )
        self._trace_routing(updated)
        return updated

    async def execute_next(self, state: CaseLensAgentState, payload=None):
        if state.next_action == "human_review":
            if self.handoff_builder is None:
                from api.human_handoff import HumanReviewHandoffBuilder

                self.handoff_builder = HumanReviewHandoffBuilder()
            started = perf_counter()
            handoff = self.handoff_builder.build(state)
            if self.history_service is not None:
                await self.history_service.record_handoff(
                    state=state,
                    handoff=handoff,
                    assessment_session_id=self._history_session_id(),
                )
            self._trace_emit(
                "human_handoff",
                current_phase=state.current_phase,
                next_action=state.next_action,
                success=True,
                duration_ms=(perf_counter() - started) * 1000,
                metadata={"reason": handoff.reason},
            )
            self._trace_emit(
                "agent_completed",
                current_phase=state.current_phase,
                next_action=state.next_action,
                success=True,
                metadata={"outcome": "human_handoff"},
            )
            return handoff
        if self.tool_registry is None:
            raise RuntimeError("CaseLensAgent has no tool registry")
        from api.agent_tools import CaseLensToolSelector

        tool_name = CaseLensToolSelector.tool_name_for(state)
        if tool_name is None:
            return None
        if payload is None:
            if tool_name != "selective_rerun":
                raise ValueError(f"Tool {tool_name} requires an explicit request payload")
            payload = state
        is_rerun = tool_name == "selective_rerun"
        if is_rerun:
            if self.history_service is not None:
                await self.history_service.record_rerun_started(
                    state,
                    self._history_session_id(),
                )
            self._trace_emit(
                "rerun_started", current_phase=state.current_phase,
                next_action=state.next_action, tool_name=tool_name,
            )
        self._trace_emit(
            "tool_started", current_phase=state.current_phase,
            next_action=state.next_action, tool_name=tool_name,
        )
        started = perf_counter()
        try:
            result = await self.tool_registry.get(tool_name).execute(payload)
        except Exception as error:
            duration = (perf_counter() - started) * 1000
            self._trace_emit(
                "tool_failed", current_phase=state.current_phase,
                next_action=state.next_action, tool_name=tool_name,
                success=False, duration_ms=duration, error=str(error),
            )
            raise
        duration = (perf_counter() - started) * 1000
        self._trace_emit(
            "tool_completed" if result.success else "tool_failed",
            current_phase=state.current_phase,
            next_action=state.next_action,
            tool_name=tool_name,
            success=result.success,
            duration_ms=duration,
            error=result.error,
        )
        if is_rerun:
            rerun_state = (
                result.output
                if result.success and isinstance(result.output, CaseLensAgentState)
                else None
            )
            rerun_events = []
            if self.history_service is not None:
                rerun_events = await self.history_service.record_rerun_completed(
                    before=state,
                    after=rerun_state,
                    success=result.success,
                    error=result.error,
                    assessment_session_id=self._history_session_id(),
                )
            if (
                self.assessment_version_service is not None
                and rerun_state is not None
                and rerun_state.rerun_failure_count == state.rerun_failure_count
                and self.assessment_version_service.is_stable_assessment(rerun_state)
            ):
                trigger_reference_id = next(
                    (
                        event.event_id
                        for event in rerun_events
                        if event.event_type == "selective_rerun_completed"
                    ),
                    None,
                )
                await self.assessment_version_service.create_version(
                    rerun_state,
                    trigger_type="selective_rerun",
                    trigger_reference_id=trigger_reference_id,
                    rerun_from_phase=state.current_phase,
                )
            self._trace_emit(
                "rerun_completed", current_phase=state.current_phase,
                next_action=state.next_action, tool_name=tool_name,
                success=result.success, duration_ms=duration, error=result.error,
            )
        if result.success and isinstance(result.output, CaseLensAgentState):
            self._trace_routing(result.output)
            self._trace_terminal_state(result.output)
        return result

    def _trace_routing(self, state: CaseLensAgentState) -> None:
        self._trace_emit(
            "routing_decision",
            current_phase=state.current_phase,
            next_action=state.next_action,
            success=True,
        )

    def _trace_terminal_state(self, state: CaseLensAgentState) -> None:
        if state.next_action == "ask_for_information":
            questions = MissingInformationInteraction.questions(state)
            self._trace_emit(
                "question_requested",
                current_phase=state.current_phase,
                next_action=state.next_action,
                success=True,
                metadata={
                    "question_count": len(questions),
                    "missing_information_ids": [str(item.id) for item in questions],
                },
            )
        elif state.next_action == "completed":
            self._trace_emit(
                "agent_completed",
                current_phase=state.current_phase,
                next_action=state.next_action,
                success=True,
                metadata={"outcome": "completed"},
            )

    def _trace_emit(self, event_type, **kwargs) -> None:
        try:
            self.trace_collector.emit(event_type, **kwargs)
        except Exception:
            return

    def _history_session_id(self) -> uuid.UUID | None:
        trace = self.trace
        return trace.trace_id if trace is not None else None


class SelectiveRerunner:
    _PHASES = (
        "fact_validation",
        "coverage_assessment",
        "exclusion_assessment",
        "obligation_assessment",
        "claim_calculation",
    )
    _REQUIRED_FOR_PHASE = {
        "fact_validation": "fact_validation",
        "policy_applicability": "coverage_assessment",
        "coverage_assessment": "coverage_assessment",
        "exclusion_assessment": "exclusion_assessment",
        "obligation_assessment": "obligation_assessment",
        "claim_calculation": "claim_calculation",
    }

    def __init__(
        self,
        *,
        validation_service,
        coverage_service,
        exclusion_service,
        obligation_service,
        calculation_service,
        recommendation_service,
        router: CaseLensAgentRouter | None = None,
    ):
        self.validation = validation_service
        self.coverage = coverage_service
        self.exclusion = exclusion_service
        self.obligation = obligation_service
        self.calculation = calculation_service
        self.recommendation = recommendation_service
        self.router = router or CaseLensAgentRouter()

    async def rerun(self, state: CaseLensAgentState) -> CaseLensAgentState:
        if state.next_action != "rerun_phase":
            raise ValueError("The agent state is not scheduled for a phase rerun")
        if state.current_phase not in self._PHASES:
            raise ValueError(f"Unsupported selective rerun phase: {state.current_phase}")
        if state.claim is None or state.knowledge_base_id is None:
            raise ValueError("Claim and knowledge_base_id are required for selective rerun")

        start = self._PHASES.index(state.current_phase)
        replaced_phases = set(self._PHASES[start:])
        working = state.model_copy(deep=True)
        working.missing_information = [
            item for item in working.missing_information
            if self._REQUIRED_FOR_PHASE.get(item.required_for) not in replaced_phases
        ]
        self._invalidate_downstream(working, start)

        try:
            if start <= 0:
                working.current_phase = "fact_validation"
                validation = self.validation.validate(FactValidationRequest(
                    claim=working.claim,
                    facts=working.facts,
                    claimant_reference_required=working.claimant_reference_required,
                    policy_reference_required=working.policy_reference_required,
                ))
                working.missing_information.extend(validation.missing_information)
                if not validation.ready_for_assessment:
                    return self._finish_early_with_recommendation(working)

            if start <= 1:
                working.current_phase = "coverage_assessment"
                coverage = await self.coverage.assess(CoverageAssessmentRequest(
                    knowledge_base_id=working.knowledge_base_id,
                    claim=working.claim,
                    facts=working.facts,
                    retrieval_limit=working.retrieval_limit,
                ))
                working.applicable_policy = coverage.applicable_policy
                working.coverage_assessments = coverage.coverage_assessments
                working.missing_information.extend(coverage.missing_information)
                if not working.coverage_assessments:
                    return self._finish_early_with_recommendation(working)

            if start <= 2:
                working.current_phase = "exclusion_assessment"
                exclusions = await self.exclusion.assess(ExclusionAssessmentRequest(
                    knowledge_base_id=working.knowledge_base_id,
                    claim=working.claim,
                    facts=working.facts,
                    coverage_assessments=working.coverage_assessments,
                    retrieval_limit=working.retrieval_limit,
                ))
                working.exclusion_assessments = exclusions.exclusion_assessments
                working.missing_information.extend(exclusions.missing_information)

            if start <= 3:
                working.current_phase = "obligation_assessment"
                obligations = await self.obligation.assess(ObligationAssessmentRequest(
                    knowledge_base_id=working.knowledge_base_id,
                    claim=working.claim,
                    facts=working.facts,
                    coverage_assessments=working.coverage_assessments,
                    exclusion_assessments=working.exclusion_assessments,
                    retrieval_limit=working.retrieval_limit,
                ))
                working.obligation_assessments = obligations.obligation_assessments
                working.missing_information.extend(obligations.missing_information)

            if start <= 4:
                working.current_phase = "claim_calculation"
                calculations = await self.calculation.calculate(ClaimCalculationRequest(
                    knowledge_base_id=working.knowledge_base_id,
                    claim=working.claim,
                    facts=working.facts,
                    coverage_assessments=working.coverage_assessments,
                    exclusion_assessments=working.exclusion_assessments,
                    obligation_assessments=working.obligation_assessments,
                    retrieval_limit=working.retrieval_limit,
                ))
                working.calculation_results = calculations.calculation_results
                working.missing_information.extend(calculations.missing_information)

            return self._finish_with_recommendation(working)
        except Exception as error:
            working.rerun_failure_count += 1
            if "Provider" in type(error).__name__:
                working.provider_errors.append(f"{type(error).__name__}: {error}")
            working.next_action = (
                "human_review" if working.rerun_failure_count >= 2 else "rerun_phase"
            )
            return working

    @staticmethod
    def _invalidate_downstream(state: CaseLensAgentState, start: int) -> None:
        if start <= 1:
            state.applicable_policy = None
            state.coverage_assessments = []
        if start <= 2:
            state.exclusion_assessments = []
        if start <= 3:
            state.obligation_assessments = []
        if start <= 4:
            state.calculation_results = []
        state.recommendation = None

    def _finish_early_with_recommendation(
        self, state: CaseLensAgentState
    ) -> CaseLensAgentState:
        blocked_phase = state.current_phase
        result = self._recommend(state)
        state.recommendation = result.recommendation
        state.current_phase = blocked_phase
        state.next_action = "continue_assessment"
        self._deduplicate(state)
        return self.router.route(state)

    def _finish_with_recommendation(self, state: CaseLensAgentState) -> CaseLensAgentState:
        self._deduplicate(state)
        result = self._recommend(state)
        state.recommendation = result.recommendation
        state.current_phase = "claim_recommendation"
        state.next_action = "continue_assessment"
        return self.router.route(state)

    def _recommend(self, state: CaseLensAgentState):
        return self.recommendation.recommend(ClaimRecommendationRequest(
            claim=state.claim,
            coverage_assessments=state.coverage_assessments,
            exclusion_assessments=state.exclusion_assessments,
            obligation_assessments=state.obligation_assessments,
            calculation_results=state.calculation_results,
            missing_information=state.missing_information,
        ))

    @staticmethod
    def _deduplicate(state: CaseLensAgentState) -> None:
        from api.claim_assessment import ClaimAssessmentOrchestrator

        ClaimAssessmentOrchestrator._deduplicate_and_relink(state)
