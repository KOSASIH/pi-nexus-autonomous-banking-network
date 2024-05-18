class Network:
    """
    Represents the autonomous banking network.
    """

    def __init__(self, nodes: list):
        """
        Initializes the network with a list of nodes.

        Args:
            nodes (list): A list of node objects.
        """
        self.nodes = nodes

    def send_transaction(self, transaction: dict) -> bool:
        """
        Sends a transaction to the network.

        Args:
            transaction (dict): A dictionary representing the transaction.

        Returns:
            bool: True if the transaction was sent successfully, False otherwise.
        """
        try:
            # Send transaction to nodes
            for node in self.nodes:
                node.send_transaction(transaction)
            return True
        except Exception as e:
            print(f"Error sending transaction: {e}")
            return False
