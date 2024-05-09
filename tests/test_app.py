import unittest

class TestApp(unittest.TestCase):
    def test_database_url(self):
        self.assertIsNotNone(DATABASE_URL)

    def test_secret_key(self):
        self.assertIsNotNone(SECRET_KEY)

if __name__ == "__main__":
    unittest.main()
