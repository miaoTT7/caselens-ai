import unittest

from pgvector.sqlalchemy import VECTOR

from api.models import Chunk, Document, EMBEDDING_DIMENSION, KnowledgeBase


class PersistenceModelTests(unittest.TestCase):
    def test_expected_tables_are_registered(self):
        self.assertEqual(KnowledgeBase.__tablename__, "knowledge_bases")
        self.assertEqual(Document.__tablename__, "documents")
        self.assertEqual(Chunk.__tablename__, "chunks")

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


if __name__ == "__main__":
    unittest.main()
