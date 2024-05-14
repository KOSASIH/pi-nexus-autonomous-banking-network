class BankingNetwork:
    def __init__(self, url):
        # Initialize the banking network
        self.url = url

    def send_data(self, data):
        """
        Send data to the banking network.

        Args:
            data (dict): The data to be sent.

        Raises:
            ConnectionError: If the connection to the banking network fails.
        """
        send_data_to_server(data, self.url)

    def receive_data(self):
        """
        Receive data from the banking network.

        Raises:
            ConnectionError: If the connection to the banking network fails.
        """
        return receive_data_from_server(self.url)
