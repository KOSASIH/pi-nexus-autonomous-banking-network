class Node:
    """
    Represents a node in the autonomous banking network.
    """

    def __init__(self, node_id: int, node_address: str):
        """
        Initializes the node with a unique ID and address.

        Args:
            node_id (int): The unique node ID.
            node_address (str): The node address.
        """
        self.node_id = node_id
        self.node_address = node_address

    def send_transaction(self, transaction: dict) -> bool:
        """
        Sends a transaction to another node.

        Args:
            transaction (dict): A dictionary representing the transaction.

        Returns:
            bool: True if the transaction was sent successfully, False otherwise.
        """
        try:
            # Send transaction to another node
            # ...
            return True
        except Exception as e:
            print(f"Error sending transaction from node {self.node_id}: {e}")
            return False
