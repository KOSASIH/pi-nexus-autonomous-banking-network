# banking_app.py
import sys
import getpass
from citi_bank import CitiBank
from wells_fargo import WellsFargo
from pnc_bank import PNCBank

def main():
    api_key = getpass.getpass("Enter your API key: ")
    bank_api = {
        "Citi Bank": CitiBank(api_key),
        "Wells Fargo": WellsFargo(api_key),
        "PNC Bank": PNCBank(api_key)
    }
    while True:
        print("\nChoose a bank to interact with:")
        for index, bank in enumerate(bank_api.keys()):
            print(f"{index + 1}. {bank}")
        choice = int(input("Enter the number of your choice: "))
        bank = list(bank_api.keys())[choice - 1]
        bank_api_obj = bank_api[bank]
        while True:
            print("\nChoose an operation to perform:")
            print("1. Get accounts")
            print("2. Get transactions")
            print("3. Transfer funds")
            print("4. Go back to main menu")
            choice = int(input("Enter the number of your choice: "))
            if choice == 1:
                accounts = bank_api_obj.get_accounts()
                print("\nAccounts:")
                for account in accounts:
                    print(f"ID: {account['id']}, Name: {account['name']}, Balance: {account['balance']}")
            elif choice == 2:
                account_id = input("Enter the account ID: ")
                transactions = bank_api_obj.get_transactions(account_id)
                print("\nTransactions:")
                for transaction in transactions:
                    print(f"ID: {transaction['id']}, Date: {transaction['date']}, Amount: {transaction['amount']}")
            elif choice == 3:
                from_account_id = input("Enter the account ID to transfer from: ")
                to_account_id = input("Enter the account ID to transfer to: ")
                amount = float(input("Enter the amount to transfer: "))
                result = bank_api_obj.transfer_funds(from_account_id, to_account_id, amount)
                print("\nTransfer result:")
                print(json.dumps(result, indent=2))
            elif choice == 4:
                break
            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
