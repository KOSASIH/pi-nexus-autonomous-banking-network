import uuid


class IDManagement:
    def __init__(self):
        self.ids = set()

    def generate_id(self):
        """
        Generate a unique ID.
        """
        id = str(uuid.uuid4())
        while id in self.ids:
            id = str(uuid.uuid4())
        self.ids.add(id)
        return id

    def validate_id(self, id):
        """
        Validate a given ID.
        """
        return id in self.ids
