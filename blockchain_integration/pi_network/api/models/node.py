# node.py
from django.db import models

class Node(models.Model):
    node_id = models.CharField(max_length=255, unique=True)
    public_key = models.CharField(max_length=255, unique=True)
    address = models.CharField(max_length=255)

    def __str__(self):
        return f"Node {self.node_id}"
