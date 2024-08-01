Bank Account Manager
=====================

The Bank Account Manager module enables the management of bank accounts for fiat transactions.

**Methods**

* `create_bank_account(account_holder_name, account_number)`: Create a new bank account.
* `get_bank_account(account_number)`: Get an existing bank account.

**Usage**

```python
from bank_account_manager import BankAccountManager

bank_account_manager = BankAccountManager()
bank_account = bank_account_manager.create_bank_account("John Doe", "1234567890")
