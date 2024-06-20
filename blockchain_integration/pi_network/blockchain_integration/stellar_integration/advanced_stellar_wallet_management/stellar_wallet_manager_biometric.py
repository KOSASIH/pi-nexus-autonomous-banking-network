import hashlib
from stellar_sdk import Server, Keypair, TransactionBuilder, Network
import face_recognition
import cv2

class StellarWalletManagerBiometric:
    def __init__(self, network_passphrase, horizon_url):
        self.network_passphrase = network_passphrase
        self.horizon_url = horizon_url
        self.server = Server(horizon_url)
        self.face_recognition_model = face_recognition.FaceRecognition()

    def generate_keypair(self, seed_phrase):
        keypair = Keypair.from_secret(seed_phrase)
        return keypair

    def create_account(self, keypair, starting_balance):
        transaction = TransactionBuilder(
            source_account=keypair.public_key,
            network_passphrase=self.network_passphrase,
            base_fee=100
        ).append_create_account_op(
            destination=keypair.public_key,
            starting_balance=starting_balance
        ).build()
        self.server.submit_transaction(transaction)

    def authenticate_user(self, user_id, image_path):
        user_face_encoding = self.face_recognition_model.get_face_encoding(image_path)
        stored_face_encoding = self.face_recognition_model.get_stored_face_encoding(user_id)
        if self.face_recognition_model.compare_faces([stored_face_encoding], user_face_encoding):
            return True
        return False

    def send_payment(self, source_keypair, destination_public_key, amount, user_id, image_path):
        if self.authenticate_user(user_id, image_path):
            transaction = TransactionBuilder(
                source_account=source_keypair.public_key,
                network_passphrase=self.network_passphrase,
                base_fee=100
            ).append_payment_op(
                destination=destination_public_key,
                amount=amount,
                asset_code="XLM"
            ).build()
            self.server.submit_transaction(transaction)
        else:
            print("Authentication failed")

    def get_account_balance(self, public_key):
        account = self.server.accounts().account_id(public_key).call()
        return account.balances[0].balance
