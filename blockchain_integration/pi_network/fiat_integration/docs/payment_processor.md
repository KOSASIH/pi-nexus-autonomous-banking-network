Payment Processor
==================

The Payment Processor module enables the processing of fiat transactions.

**Methods**

* `process_payment(user_id, amount_pi, fiat_currency, recipient_address)`: Process a fiat payment.

**Usage**

```python
from payment_processor import PaymentProcessor

payment_processor = PaymentProcessor()
transaction_id = payment_processor.process_payment("user123", 100, "USD", "recipient_address")
