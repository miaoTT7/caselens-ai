"""Read-only restoration and deterministic routing of persisted claim workflows."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field, ValidationError

from api.agent import (
    CaseLensAgentRouter,
    CaseLensAgentState,
    MissingInformationInteraction,
    NextAction,
    PhaseName,
)
from api.agent_state_persistence import (
    ClaimAgentStatePersistenceService,
    UnsupportedAgentStateSchemaError,
)
from api.agent_tools import CaseLensToolSelector
from api.claim_schemas import MissingInformation
from api.human_handoff import HumanReviewHandoff, HumanReviewHandoffBuilder


class ResumeStateNotFoundError(LookupError):
    pass


class ResumeSchemaVersionError(ValueError):
    pass


class ResumeStateValidationError(ValueError):
    pass


class ResumeResult(BaseModel):
    claim_id: uuid.UUID
    state: CaseLensAgentState
    revision: int = Field(ge=1)
    state_schema_version: int = Field(ge=1)
    current_phase: PhaseName | None = None
    next_action: NextAction
    blocking_questions: list[MissingInformation] = Field(default_factory=list)
    target_phase: PhaseName | None = None
    next_tool_name: str | None = None
    human_handoff: HumanReviewHandoff | None = None


class ClaimResumeService:
    """Load and describe the next workflow step without executing it."""

    def __init__(
        self,
        persistence_service: ClaimAgentStatePersistenceService,
        *,
        router: CaseLensAgentRouter | None = None,
        handoff_builder: HumanReviewHandoffBuilder | None = None,
    ):
        self.persistence_service = persistence_service
        self.router = router or CaseLensAgentRouter()
        self.handoff_builder = handoff_builder or HumanReviewHandoffBuilder()

    async def resume(self, claim_id: uuid.UUID) -> ResumeResult:
        try:
            persisted = await self.persistence_service.load_latest_state(claim_id)
        except UnsupportedAgentStateSchemaError as error:
            raise ResumeSchemaVersionError(str(error)) from error
        except (ValidationError, ValueError, TypeError) as error:
            raise ResumeStateValidationError(
                f"Persisted state for claim {claim_id} is invalid"
            ) from error

        if persisted is None:
            raise ResumeStateNotFoundError(
                f"No persisted agent state for claim {claim_id}"
            )

        try:
            raw_state = persisted.state
            payload = (
                raw_state.model_dump(mode="python")
                if isinstance(raw_state, CaseLensAgentState)
                else raw_state
            )
            restored = CaseLensAgentState.model_validate(payload)
            if restored.claim is None:
                raise ValueError("Persisted state has no claim")
            if restored.claim.id != claim_id:
                raise ValueError("Persisted state claim.id does not match claim_id")

            routed = self.router.route(restored)
            result = self._build_result(
                claim_id=claim_id,
                state=routed,
                revision=persisted.revision,
                state_schema_version=persisted.state_schema_version,
            )
        except (ValidationError, ValueError, TypeError) as error:
            raise ResumeStateValidationError(
                f"Persisted state for claim {claim_id} is invalid"
            ) from error
        return result

    def _build_result(
        self,
        *,
        claim_id: uuid.UUID,
        state: CaseLensAgentState,
        revision: int,
        state_schema_version: int,
    ) -> ResumeResult:
        questions: list[MissingInformation] = []
        target_phase: PhaseName | None = None
        next_tool_name: str | None = None
        handoff: HumanReviewHandoff | None = None

        if state.next_action == "ask_for_information":
            questions = MissingInformationInteraction.questions(state)
        elif state.next_action == "rerun_phase":
            if state.current_phase is None:
                raise ValueError("rerun_phase requires current_phase")
            target_phase = state.current_phase
        elif state.next_action == "human_review":
            handoff = self.handoff_builder.build(state)
        elif state.next_action == "completed":
            if state.recommendation is None:
                raise ValueError("completed state requires recommendation")
        elif state.next_action == "continue_assessment":
            next_tool_name = CaseLensToolSelector.tool_name_for(state)

        return ResumeResult(
            claim_id=claim_id,
            state=state,
            revision=revision,
            state_schema_version=state_schema_version,
            current_phase=state.current_phase,
            next_action=state.next_action,
            blocking_questions=questions,
            target_phase=target_phase,
            next_tool_name=next_tool_name,
            human_handoff=handoff,
        )
