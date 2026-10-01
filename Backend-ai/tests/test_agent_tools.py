import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from api.agent import CaseLensAgent, CaseLensAgentState
from api.agent_tools import (
    CaseLensToolSelector,
    ServiceTool,
    ToolRegistry,
    build_case_lens_tool_registry,
)


class ToolRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_registry_registers_lists_and_executes_sync_or_async_tools(self):
        registry = ToolRegistry()
        registry.register(ServiceTool(
            name="sync_tool", description="Sync tool.",
            affected_phase="fact_validation", execute=lambda value: value + 1,
        ))
        async_execute = AsyncMock(return_value="done")
        registry.register(ServiceTool(
            name="async_tool", description="Async tool.",
            affected_phase="coverage_assessment", execute=async_execute,
        ))

        sync_result = await registry.get("sync_tool").execute(1)
        async_result = await registry.get("async_tool").execute("request")

        self.assertTrue(sync_result.success)
        self.assertEqual(sync_result.output, 2)
        self.assertEqual(async_result.output, "done")
        self.assertEqual(
            [item.name for item in registry.list_metadata()],
            ["sync_tool", "async_tool"],
        )

    async def test_tool_failure_returns_envelope_without_invented_output(self):
        tool = ServiceTool(
            name="failing", description="Fails.", execute=Mock(side_effect=RuntimeError("boom")),
        )

        result = await tool.execute({})

        self.assertFalse(result.success)
        self.assertIsNone(result.output)
        self.assertEqual(result.error, "RuntimeError: boom")


class CaseLensToolMappingTests(unittest.IsolatedAsyncioTestCase):
    def build_registry(self):
        validation = SimpleNamespace(validate=Mock(return_value="validated"))
        coverage = SimpleNamespace(assess=AsyncMock(return_value="coverage"))
        exclusion = SimpleNamespace(assess=AsyncMock(return_value="exclusions"))
        obligation = SimpleNamespace(assess=AsyncMock(return_value="obligations"))
        calculation = SimpleNamespace(calculate=AsyncMock(return_value="calculation"))
        recommendation = SimpleNamespace(recommend=Mock(return_value="recommendation"))
        rerunner = SimpleNamespace(rerun=AsyncMock(return_value="rerun"))
        registry = build_case_lens_tool_registry(
            validation_service=validation,
            coverage_service=coverage,
            exclusion_service=exclusion,
            obligation_service=obligation,
            calculation_service=calculation,
            recommendation_service=recommendation,
            selective_rerunner=rerunner,
        )
        return registry, validation, coverage, rerunner

    async def test_registry_tools_delegate_to_existing_services(self):
        registry, validation, coverage, _ = self.build_registry()

        validation_result = await registry.get("validate_facts").execute("validation request")
        coverage_result = await registry.get("assess_coverage").execute("coverage request")

        validation.validate.assert_called_once_with("validation request")
        coverage.assess.assert_awaited_once_with("coverage request")
        self.assertEqual(validation_result.output, "validated")
        self.assertEqual(coverage_result.output, "coverage")
        self.assertEqual(len(registry.list_metadata()), 7)

    def test_selector_is_deterministic(self):
        rerun = CaseLensAgentState(
            current_phase="exclusion_assessment", next_action="rerun_phase"
        )
        continued = CaseLensAgentState(
            current_phase="claim_calculation", next_action="continue_assessment"
        )
        waiting = CaseLensAgentState(next_action="ask_for_information")

        self.assertEqual(CaseLensToolSelector.tool_name_for(rerun), "selective_rerun")
        self.assertEqual(CaseLensToolSelector.tool_name_for(continued), "calculate_claim")
        self.assertIsNone(CaseLensToolSelector.tool_name_for(waiting))

    async def test_agent_executes_only_deterministically_selected_tool(self):
        registry, _, _, rerunner = self.build_registry()
        state = CaseLensAgentState(
            knowledge_base_id=uuid.uuid4(),
            current_phase="obligation_assessment",
            next_action="rerun_phase",
        )
        agent = CaseLensAgent(AsyncMock(), tool_registry=registry)

        result = await agent.execute_next(state)

        rerunner.rerun.assert_awaited_once_with(state)
        self.assertTrue(result.success)
        self.assertEqual(result.tool_name, "selective_rerun")
        self.assertEqual(result.affected_phase, "obligation_assessment")


if __name__ == "__main__":
    unittest.main()
