import json
import os
import socket

from cryptography.fernet import Fernet


class MicrotransactionHandler:

    def __init__(self, bank_account, microtransaction_secret):
        self.bank_account = bank_account
        self.microtransaction_secret = microtransaction_secret
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(("localhost", 8081))
        self.socket.listen(5)
        self.threads = []

    def handle_microtransaction(self, client_socket):
        data = client_socket.recv(1024)
        if data:
            transaction_data = json.loads(data.decode())
            if transaction_data["type"] == "TRANSFER":
                amount = float(transaction_data["amount"])
                if amount <= self.bank_account["balance"]:
                    self.bank_account["balance"] -= amount
                    print(
                        f'Transfer successful. New balance: {self.bank_account["balance"]}'
                    )
                    client_socket.send("TRANSFER_SUCCESSFUL".encode())
                else:
                    print("Insufficient funds")
                    client_socket.send("INSUFFICIENT_FUNDS".encode())
            elif transaction_data["type"] == "BALANCE":
                client_socket.send(str(self.bank_account["balance"]).encode())
        else:
            client_socket.close()

    def start_listening(self):
        while True:
            client_socket, address = self.socket.accept()
            thread = threading.Thread(
                target=self.handle_microtransaction, args=(client_socket,)
            )
            thread.start()
            self.threads.append(thread)

    def authenticate_client(self, client_socket):
        cipher_suite = Fernet(self.microtransaction_secret)
        encrypted_client_secret = client_socket.recv(1024)
        if encrypted_client_secret == cipher_suite.encrypt(
            b"microtransaction_client_secret"
        ):
            return True
        else:
            return False

    def start(self):
        thread = threading.Thread(target=self.start_listening)
        thread.start()
        self.threads.append(thread)

    def stop(self):
        for thread in self.threads:
            thread.join()


if __name__ == "__main__":
    bank_account = {"balance": 1000.0}
    microtransaction_secret = "microtransaction_secret123"
    microtransaction_handler = MicrotransactionHandler(
        bank_account, microtransaction_secret
    )
    microtransaction_handler.start()
