"""Framework-free tool adapters for existing CaseLens services."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel

from api.agent import CaseLensAgentState, NextAction, PhaseName


class ToolMetadata(BaseModel):
    name: str
    description: str
    affected_phase: PhaseName | None = None


class ToolExecutionResult(BaseModel):
    tool_name: str
    success: bool
    output: Any | None = None
    error: str | None = None
    affected_phase: PhaseName | None = None


class CaseLensTool(Protocol):
    name: str
    description: str
    affected_phase: PhaseName | None

    async def execute(self, payload: Any) -> ToolExecutionResult: ...


class ServiceTool:
    def __init__(
        self,
        *,
        name: str,
        description: str,
        execute: Callable[[Any], Any],
        affected_phase: PhaseName | None = None,
        phase_resolver: Callable[[Any], PhaseName | None] | None = None,
    ):
        self.name = name
        self.description = description
        self.affected_phase = affected_phase
        self._execute = execute
        self._phase_resolver = phase_resolver

    async def execute(self, payload: Any) -> ToolExecutionResult:
        phase = self._phase_resolver(payload) if self._phase_resolver else self.affected_phase
        try:
            output = self._execute(payload)
            if inspect.isawaitable(output):
                output = await output
            return ToolExecutionResult(
                tool_name=self.name,
                success=True,
                output=output,
                affected_phase=phase,
            )
        except Exception as error:
            return ToolExecutionResult(
                tool_name=self.name,
                success=False,
                error=f"{type(error).__name__}: {error}",
                affected_phase=phase,
            )


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, CaseLensTool] = {}

    def register(self, tool: CaseLensTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool is already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> CaseLensTool:
        try:
            return self._tools[name]
        except KeyError as error:
            raise LookupError(f"Unknown tool: {name}") from error

    def list_metadata(self) -> list[ToolMetadata]:
        return [
            ToolMetadata(
                name=tool.name,
                description=tool.description,
                affected_phase=tool.affected_phase,
            )
            for tool in self._tools.values()
        ]


class CaseLensToolSelector:
    _TOOL_BY_PHASE: dict[PhaseName, str] = {
        "fact_validation": "validate_facts",
        "coverage_assessment": "assess_coverage",
        "exclusion_assessment": "assess_exclusions",
        "obligation_assessment": "assess_obligations",
        "claim_calculation": "calculate_claim",
        "claim_recommendation": "generate_recommendation",
    }

    @classmethod
    def tool_name_for(cls, state: CaseLensAgentState) -> str | None:
        if state.next_action == "rerun_phase":
            return "selective_rerun"
        if state.next_action != "continue_assessment" or state.current_phase is None:
            return None
        return cls._TOOL_BY_PHASE.get(state.current_phase)


def build_case_lens_tool_registry(
    *,
    validation_service,
    coverage_service,
    exclusion_service,
    obligation_service,
    calculation_service,
    recommendation_service,
    selective_rerunner,
) -> ToolRegistry:
    registry = ToolRegistry()
    definitions = (
        ("validate_facts", "Validate basic claim completeness.",
         validation_service.validate, "fact_validation", None),
        ("assess_coverage", "Assess applicable policy coverage.",
         coverage_service.assess, "coverage_assessment", None),
        ("assess_exclusions", "Assess relevant policy exclusions.",
         exclusion_service.assess, "exclusion_assessment", None),
        ("assess_obligations", "Assess policyholder obligations.",
         obligation_service.assess, "obligation_assessment", None),
        ("calculate_claim", "Calculate claim amounts deterministically.",
         calculation_service.calculate, "claim_calculation", None),
        ("generate_recommendation", "Generate the deterministic claim recommendation.",
         recommendation_service.recommend, "claim_recommendation", None),
        ("selective_rerun", "Rerun the earliest affected phase and its dependencies.",
         selective_rerunner.rerun, None, lambda state: state.current_phase),
    )
    for name, description, execute, phase, resolver in definitions:
        registry.register(ServiceTool(
            name=name,
            description=description,
            execute=execute,
            affected_phase=phase,
            phase_resolver=resolver,
        ))
    return registry
