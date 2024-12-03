import unittest
from identity import IdentityManager  # Assuming you have an IdentityManager class

class TestIdentityManager(unittest.TestCase):
    def setUp(self):
        self.identity_manager = IdentityManager()

    def test_create_identity(self):
        identity = self.identity_manager.create_identity("Alice", "alice@example.com")
        self.assertIsNotNone(identity)
        self.assertEqual(identity.name, "Alice")

    def test_get_identity(self):
        self.identity_manager.create_identity("Bob", "bob@example.com")
        identity = self.identity_manager.get_identity("Bob")
        self.assertEqual(identity.email, "bob@example.com")

    def test_identity_not_found(self):
        identity = self.identity_manager.get_identity("NonExistent")
        self.assertIsNone(identity)

if __name__ == "__main__":
    unittest.main()
