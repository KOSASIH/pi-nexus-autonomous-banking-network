import argparse
import os
import sys
from api import API
from cryptography import Cryptography

def get_user_input(prompt):
    return input(f'{prompt}: ')

def print_error(message):
    print(f'\033[91m{message}\033[0m')

def print_success(message):
    print(f'\033[92m{message}\033[0m')

def main():
    parser = argparse.ArgumentParser(description='PiNexus Autonomous Banking Network CLI')
    parser.add_argument('--api-key', required=True, help='API key')
    parser.add_argument('--api-secret', required=True, help='API secret')
    parser.add_argument('--host', default='localhost', help='API host')
    parser.add_argument('--port', default=5000, type=int, help='API port')
    parser.add_argument('--command', choices=['balance', 'transfer', 'generate-address', 'transaction', 'transactions', 'sign', 'verify', 'encrypt', 'decrypt'], required=True, help='Command to execute')
    parser.add_argument('--user-id', help='User ID')
    parser.add_argument('--receiver-id', help='Receiver ID')
    parser.add_argument('--amount', type=float, help='Amount')
    parser.add_argument('--transaction-id', help='Transaction ID')
    parser.add_argument('--limit', type=int, default=10, help='Limit')
    parser.add_argument('--offset', type=int, default=0, help='Offset')
    parser.add_argument('--message', help='Message')
    parser.add_argument('--signature', help='Signature')
    parser.add_argument('--ciphertext', help='Ciphertext')
    args = parser.parse_args()

    api = API(args.host, args.port, args.api_key, args.api_secret)
    cryptography = Cryptography()

    if args.command == 'balance':
        if not args.user_id:
            print_error('User ID is required')
            sys.exit(1)

        balance = api.get_balance(args.user_id)
        print_success(f'Balance: {balance}')

    elif args.command == 'transfer':
        if not args.user_id or not args.receiver_id or not args.amount:
            print_error('User ID, Receiver ID, and Amount are required')
            sys.exit(1)

        transaction_id = api.transfer(args.user_id, args.receiver_id, args.amount)
        print_success(f'Transaction ID: {transaction_id}')

    elif args.command == 'generate-address':
        if not args.user_id:
            print_error('User ID is required')
            sys.exit(1)

        address = api.generate_address(args.user_id)
        print_success(f'Address: {address}')

    elif args.command == 'transaction':
        if not args.transaction_id:
            print_error('Transaction ID is required')
            sys.exit(1)

        transaction = api.get_transaction(args.transaction_id)
        print_success(json.dumps(transaction, indent=4))

    elif args.command == 'transactions':
        if not args.user_id:
            print_error('User ID is required')
            sys.exit(1)

        transactions = api.get_transactions(args.user_id, args.limit, args.offset)
        print_success(json.dumps(transactions, indent=4))

    elif args.command == 'sign':
        if not args.message:
            print_error('Message is required')
            sys.exit(1)

        signature = cryptography.sign_message(args.message)
        print_success(f'Signature: {signature}')

elif args.command == 'verify':
        if not args.message or not args.signature:
            print_error('Message and Signature are required')
            sys.exit(1)

        is_valid = cryptography.verify_message(args.message, args.signature)
        print_success(f'Is valid: {is_valid}')

    elif args.command == 'encrypt':
        if not args.message or not args.receiver_id:
            print_error('Message and Receiver ID are required')
            sys.exit(1)

        ciphertext = api.encrypt_message(args.message, args.receiver_id)
        print_success(f'Ciphertext: {ciphertext}')

    elif args.command == 'decrypt':
        if not args.ciphertext or not args.sender_id:
            print_error('Ciphertext and Sender ID are required')
            sys.exit(1)

        plaintext = api.decrypt_message(args.ciphertext, args.sender_id)
        print_success(f'Plaintext: {plaintext}')

if __name__ == '__main__':
    main()
