import unittest

from pgvector.sqlalchemy import VECTOR

from sqlalchemy.dialects.postgresql import JSONB

from api.models import (
    AssessmentVersionRecord,
    ClaimAgentStateRecord,
    ClaimEventRecord,
    Chunk,
    Document,
    EMBEDDING_DIMENSION,
    KnowledgeBase,
)


class PersistenceModelTests(unittest.TestCase):
    def test_expected_tables_are_registered(self):
        self.assertEqual(KnowledgeBase.__tablename__, "knowledge_bases")
        self.assertEqual(Document.__tablename__, "documents")
        self.assertEqual(Chunk.__tablename__, "chunks")
        self.assertEqual(ClaimAgentStateRecord.__tablename__, "claim_agent_states")
        self.assertEqual(ClaimEventRecord.__tablename__, "claim_events")
        self.assertEqual(AssessmentVersionRecord.__tablename__, "assessment_versions")

    def test_chunk_embedding_uses_expected_vector_dimension(self):
        embedding_type = Chunk.__table__.c.embedding.type
        self.assertIsInstance(embedding_type, VECTOR)
        self.assertEqual(embedding_type.dim, EMBEDDING_DIMENSION)
        self.assertEqual(EMBEDDING_DIMENSION, 384)

    def test_chunk_keeps_structured_source_fields(self):
        columns = set(Chunk.__table__.columns.keys())
        self.assertTrue(
            {
                "section_path",
                "source_block_orders",
                "page_numbers",
                "positions",
                "sheet",
                "row_start",
                "row_end",
                "email_metadata",
                "context_above",
                "context_below",
            }.issubset(columns)
        )

    def test_agent_state_uses_normalized_control_and_jsonb_snapshot_fields(self):
        table = ClaimAgentStateRecord.__table__
        self.assertTrue(
            {
                "claim_id",
                "knowledge_base_id",
                "current_phase",
                "next_action",
                "revision",
                "state_schema_version",
                "rerun_failure_count",
            }.issubset(table.columns.keys())
        )
        for name in (
            "claim",
            "facts",
            "missing_information",
            "applicable_policy",
            "coverage_assessments",
            "exclusion_assessments",
            "obligation_assessments",
            "calculation_results",
            "recommendation",
            "completed_phases",
            "provider_errors",
        ):
            self.assertIsInstance(table.c[name].type, JSONB)

    def test_claim_event_history_is_structured_for_append_only_events(self):
        table = ClaimEventRecord.__table__
        self.assertTrue(
            {
                "event_id",
                "claim_id",
                "assessment_session_id",
                "event_type",
                "actor_type",
                "field_path",
                "old_value",
                "new_value",
                "raw_answer",
                "normalized_value",
                "missing_information_id",
                "evidence_refs",
                "related_phase",
                "created_at",
            }.issubset(table.columns.keys())
        )
        self.assertIsInstance(table.c.old_value.type, JSONB)
        self.assertIsInstance(table.c.normalized_value.type, JSONB)

    def test_assessment_versions_keep_immutable_snapshot_fields(self):
        table = AssessmentVersionRecord.__table__
        self.assertTrue(
            {
                "assessment_version_id",
                "claim_id",
                "parent_version_id",
                "version_number",
                "trigger_type",
                "trigger_reference_id",
                "rerun_from_phase",
                "completed_phases",
                "phase_outputs",
                "recommendation",
                "status",
                "created_at",
            }.issubset(table.columns.keys())
        )
        self.assertIsInstance(table.c.phase_outputs.type, JSONB)
        self.assertIsInstance(table.c.recommendation.type, JSONB)


if __name__ == "__main__":
    unittest.main()
