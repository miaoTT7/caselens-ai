import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from api.agent import (
    CaseLensAgent,
    CaseLensAgentState,
    MissingInformationAnswer,
    MissingInformationAnswerRequest,
)
from api.agent_tools import ServiceTool, ToolRegistry
from api.agent_tracing import TraceCollector
from api.claim_schemas import (
    Claim,
    ClaimAssessmentRequest,
    ClaimAssessmentResponse,
    ClaimRecommendation,
    MissingInformation,
)


class AgentTracingTests(unittest.IsolatedAsyncioTestCase):
    async def test_normal_run_records_start_routing_and_completion(self):
        response = ClaimAssessmentResponse(
            status="completed",
            completed_phases=["claim_recommendation"],
            claim=Claim(id=uuid.uuid4()),
            recommendation=ClaimRecommendation(
                id=uuid.uuid4(), status="recommend_approve",
                reasons=["Complete."], human_review_required=False,
                disclaimer="Decision support only.",
            ),
        )
        orchestrator = SimpleNamespace(assess=AsyncMock(return_value=response))
        agent = CaseLensAgent(orchestrator)

        await agent.assess(ClaimAssessmentRequest(
            fnol_text="Bicycle stolen.", knowledge_base_id=uuid.uuid4()
        ))

        events = agent.trace.events
        self.assertEqual(
            [item.event_type for item in events],
            ["agent_started", "routing_decision", "agent_completed"],
        )
        self.assertEqual(len({item.trace_id for item in events}), 1)

    async def test_rerun_tool_records_tool_and_rerun_lifecycle(self):
        output = CaseLensAgentState(
            claim=Claim(id=uuid.uuid4()),
            current_phase="claim_recommendation",
            next_action="completed",
        )
        registry = ToolRegistry()
        registry.register(ServiceTool(
            name="selective_rerun",
            description="Rerun.",
            execute=AsyncMock(return_value=output),
            phase_resolver=lambda state: state.current_phase,
        ))
        agent = CaseLensAgent(AsyncMock(), tool_registry=registry)
        state = CaseLensAgentState(
            claim=output.claim,
            current_phase="claim_calculation",
            next_action="rerun_phase",
        )

        result = await agent.execute_next(state)

        self.assertTrue(result.success)
        event_types = [item.event_type for item in agent.trace.events]
        self.assertEqual(event_types[:3], ["rerun_started", "tool_started", "tool_completed"])
        self.assertIn("rerun_completed", event_types)
        self.assertIn("routing_decision", event_types)
        self.assertIn("agent_completed", event_types)

    async def test_failed_tool_error_is_sanitized(self):
        registry = ToolRegistry()
        registry.register(ServiceTool(
            name="calculate_claim",
            description="Calculate.",
            execute=Mock(side_effect=RuntimeError("authorization: secret-token")),
            affected_phase="claim_calculation",
        ))
        agent = CaseLensAgent(AsyncMock(), tool_registry=registry)
        state = CaseLensAgentState(
            current_phase="claim_calculation",
            next_action="continue_assessment",
        )

        result = await agent.execute_next(state, payload="request")

        self.assertFalse(result.success)
        failed = agent.trace.events[-1]
        self.assertEqual(failed.event_type, "tool_failed")
        self.assertNotIn("secret-token", failed.error)
        self.assertIn("[REDACTED]", failed.error)

    async def test_question_and_answer_events_use_existing_interaction(self):
        missing = MissingInformation(
            id=uuid.uuid4(), field_path="incidents[0].location",
            reason="Location required", required_for="exclusion_assessment",
            question="Where did it occur?", blocking=True,
        )
        agent = CaseLensAgent(AsyncMock())
        state = CaseLensAgentState(
            missing_information=[missing], next_action="ask_for_information"
        )
        agent._trace_terminal_state(state)

        updated = await agent.apply_missing_information_answers(
            state,
            MissingInformationAnswerRequest(answers=[MissingInformationAnswer(
                missing_information_id=missing.id,
                value="outside Zurich station",
            )]),
        )

        self.assertEqual(updated.next_action, "rerun_phase")
        self.assertEqual(
            [item.event_type for item in agent.trace.events],
            ["question_requested", "user_answer_applied", "routing_decision"],
        )

    async def test_tracing_failure_does_not_break_assessment(self):
        class BrokenCollector:
            def start_trace(self):
                raise RuntimeError("trace unavailable")

            def emit(self, *args, **kwargs):
                raise RuntimeError("trace unavailable")

            def snapshot(self):
                raise RuntimeError("trace unavailable")

        response = ClaimAssessmentResponse(
            status="partial", claim=Claim(id=uuid.uuid4())
        )
        orchestrator = SimpleNamespace(assess=AsyncMock(return_value=response))
        agent = CaseLensAgent(orchestrator, trace_collector=BrokenCollector())

        state = await agent.assess(ClaimAssessmentRequest(
            fnol_text="Bicycle stolen.", knowledge_base_id=uuid.uuid4()
        ))

        self.assertEqual(state.claim.id, response.claim.id)
        self.assertIsNone(agent.trace)


class TraceCollectorTests(unittest.TestCase):
    def test_sanitizes_credentials_and_caps_error_length(self):
        error = "api_key=secret authorization: token Bearer abc.def " + ("x" * 700)
        sanitized = TraceCollector.sanitize_error(error)

        self.assertNotIn("secret", sanitized)
        self.assertNotIn("abc.def", sanitized)
        self.assertLessEqual(len(sanitized), 500)


if __name__ == "__main__":
    unittest.main()
