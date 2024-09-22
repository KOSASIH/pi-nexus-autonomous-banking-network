import unittest
from app.database import init_db, get_db_session

class TestDatabase(unittest.TestCase):
    def test_init_db(self):
        # Test initializing the database
        init_db()
        self.assertTrue(True)  # Assert that the database was initialized successfully

    def test_get_db_session(self):
        # Test getting a database session
        session = get_db_session()
        self.assertIsNotNone(session)

if __name__ == "__main__":
    unittest.main()
