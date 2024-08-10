from models import User

class UserService:
  def __init__(self):
    self.users = []

  def get_users(self):
    return self.users

  def add_user(self, user):
    self.users.append(user)
    return self.users

  def validate_user(self, user):
    if user.username == "":
      return False
    if user.password == "":
      return False
    return True
