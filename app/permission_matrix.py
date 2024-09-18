import json

class PermissionMatrix:
    def __init__(self):
        self.permissions = {
            'admin': ['create_user', 'delete_user', 'edit_user'],
            'moderator': ['edit_user'],
            'user': ['view_profile']
        }

    def has_permission(self, user_role, permission):
        return permission in self.permissions.get(user_role, [])

    def add_permission(self, user_role, permission):
        if user_role not in self.permissions:
            self.permissions[user_role] = []
        self.permissions[user_role].append(permission)

    def remove_permission(self, user_role, permission):
        if user_role in self.permissions and permission in self.permissions[user_role]:
            self.permissions[user_role].remove(permission)

    def save_permissions(self, file_path):
        with open(file_path, 'w') as file:
            json.dump(self.permissions, file)

    def load_permissions(self, file_path):
        with open(file_path, 'r') as file:
            self.permissions = json.load(file)
