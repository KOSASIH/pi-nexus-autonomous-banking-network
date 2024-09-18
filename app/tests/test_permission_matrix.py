import unittest
from app.permission_matrix import PermissionMatrix

class TestPermissionMatrix(unittest.TestCase):
    def setUp(self):
        self.permission_matrix = PermissionMatrix()

    def test_has_permission(self):
        self.assertTrue(self.permission_matrix.has_permission('admin', 'create_user'))
        self.assertFalse(self.permission_matrix.has_permission('user', 'create_user'))

    def test_add_permission(self):
        self.permission_matrix.add_permission('user', 'create_user')
        self.assertTrue(self.permission_matrix.has_permission('user', 'create_user'))

    def test_remove_permission(self):
        self.permission_matrix.add_permission('user', 'create_user')
        self.permission_matrix.remove_permission('user', 'create_user')
        self.assertFalse(self.permission_matrix.has_permission('user', 'create_user'))

    def test_save_permissions(self):
        self.permission_matrix.add_permission('user', 'create_user')
        self.permission_matrix.save_permissions('permissions.json')
        loaded_permission_matrix = PermissionMatrix()
        loaded_permission_matrix.load_permissions('permissions.json')
        self.assertTrue(loaded_permission_matrix.has_permission('user', 'create_user'))

    def test_load_permissions(self):
        self.permission_matrix.add_permission('user', 'create_user')
        self.permission_matrix.save_permissions('permissions.json')
        loaded_permission_matrix = PermissionMatrix()
        loaded_permission_matrix.load_permissions('permissions.json')
        self.assertTrue(loaded_permission_matrix.has_permission('user', 'create_user'))

if __name__ == '__main__':
    unittest.main()
