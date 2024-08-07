import json

class IBCProtocol:
    def __init__(self, connector: IBCConnector):
        self.connector = connector

    def send_request(self, request: dict) -> dict:
        message = json.dumps(request)
        self.connector.send_message(message, self.connector.establish_connection())
        response = self.connector.receive_message(self.connector.establish_connection())
        return json.loads(response)

    def receive_request(self) -> dict:
        message = self.connector.receive_message(self.connector.establish_connection())
        return json.loads(message)

    def send_response(self, response: dict) -> None:
        message = json.dumps(response)
        self.connector.send_message(message, self.connector.establish_connection())
