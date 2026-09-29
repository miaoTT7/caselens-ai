import os
import unittest
from unittest.mock import patch

from api.database import get_database_url


class DatabaseConfigurationTests(unittest.TestCase):
    def test_database_url_is_required(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "DATABASE_URL is not configured"):
                get_database_url()

    def test_database_url_requires_asyncpg(self):
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/caselens"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "postgresql\\+asyncpg"):
                get_database_url()

    def test_database_url_accepts_asyncpg(self):
        value = "postgresql+asyncpg://user:secret@localhost:5432/caselens"
        with patch.dict(os.environ, {"DATABASE_URL": value}, clear=True):
            self.assertEqual(get_database_url(), value)


if __name__ == "__main__":
    unittest.main()
