import uuid
import time
from notification import Notification

class PaymentGateway:
    def __init__(self):
        self.payments = {}  # payment_id -> Payment
        self.notification_service = Notification()

    def initiate_payment(self, amount, currency, payer, receiver, recurring=False, interval=None):
        payment_id = str(uuid.uuid4())
        payment = Payment(payment_id, amount, currency, payer, receiver, recurring, interval)
        self.payments[payment_id] = payment
        self.notification_service.send_notification(payer, f"Payment initiated: {payment_id} for {amount} {currency} from {payer} to {receiver}.")
        print(f"Payment initiated: {payment_id} for {amount} {currency} from {payer} to {receiver}.")
        return payment_id

    def get_payment_status(self, payment_id):
        payment = self.payments.get(payment_id)
        if payment:
            return payment.status
        return "Payment not found."

    def complete_payment(self, payment_id):
        payment = self.payments.get(payment_id)
        if payment and payment.status == "Pending":
            payment.status = "Completed"
            payment.timestamp = time.time()
            self.notification_service.send_notification(payment.receiver, f"Payment {payment_id} completed.")
            print(f"Payment {payment_id} completed.")
        else:
            print(f"Payment {payment_id} cannot be completed or does not exist.")

    def process_recurring_payments(self):
        for payment in self.payments.values():
            if payment.recurring and payment.status == "Pending":
                self.complete_payment(payment.payment_id)

class Payment:
    def __init__(self, payment_id, amount, currency, payer, receiver, recurring=False, interval=None):
        self.payment_id = payment_id
        self.amount = amount
        self.currency = currency
        self.payer = payer
        self.receiver = receiver
        self.status = "Pending"
        self.timestamp = None
        self.recurring = recurring
        self.interval = interval  # e.g., "monthly", "weekly"
