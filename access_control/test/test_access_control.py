import unittest
from access_control import AccessControl

class TestAccessControl(unittest.TestCase):
    def test_check_permissions(self):
        # Test the check_permissions method
        access_control = AccessControl()
        user_role = "admin"
        permission = "create_account"
        self.assertTrue(access_control.check_permissions(user_role, permission))

    def test_check_permissions_denied(self):
        # Test the check_permissions method with denied permission
        access_control = AccessControl()
        user_role = "user"
        permission = "delete_account"
        self.assertFalse(access_control.check_permissions(user_role, permission))

    def test_get_user_role(self):
        # Test the get_user_role method
        access_control = AccessControl()
        user_id = 1
        expected_role = "admin"
        self.assertEqual(access_control.get_user_role(user_id), expected_role)

if __name__ == "__main__":
    unittest.main()
