# tests/test_bank_of_america.py

import unittest
from unittest.mock import patch, MagicMock

from banks.bank_of_america import BankOfAmerica
from exceptions import BankIntegrationError, BankAuthenticationError, BankAPIError, BankAccountNotFoundError

class TestBankOfAmerica(unittest.TestCase):
    def setUp(self):
        self.bank_of_america = BankOfAmerica('test_username', 'test_password')

    @patch('requests.get')
    def test_get_account_info_success(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            'account_number': '1234567890',
            'account_holder': 'John Doe',
            'bank_type': 'RETAIL',
            'balance': 1000.00
        }

        account_info = self.bank_of_america.get_account_info('1234567890')

        self.assertEqual(account_info.account_number, '1234567890')
        self.assertEqual(account_info.account_holder, 'John Doe')
        self.assertEqual(account_info.bank_type, 'RETAIL')
        self.assertEqual(account_info.balance, 1000.00)

    @patch('requests.get')
    def test_get_account_info_failure(self, mock_get):
        mock_get.return_value.status_code = 404
        mock_get.return_value.json.return_value = {
            'error': 'Account not found'
        }

        with self.assertRaises(BankAccountNotFoundError):
            self.bank_of_america.get_account_info('1234567890')

    @patch('requests.post')
    def test_make_transaction_success(self, mock_post):
        mock_post.return_value.status_code = 201
        mock_post.return_value.json.return_value = {
            'transaction_id': 'abcdefg12345'
        }

        transaction_id = self.bank_of_america.make_transaction('1234567890', 100.00, '0987654321')

        self.assertEqual(transaction_id, 'abcdefg12345')

    @patch('requests.post')
    def test_make_transaction_failure(self, mock_post):
        mock_post.return_value.status_code = 400
        mock_post.return_value.json.return_value = {
            'error': 'Invalid transaction'
        }

        with self.assertRaises(BankTransactionError):
            self.bank_of_america.make_transaction('1234567890', 100.00, '0987654321')

    @patch('requests.post')
    def test_make_transaction_fraud_detected(self, mock_post):
        mock_post.return_value.status_code = 403
        mock_post.return_value.json.return_value = {
            'error': 'Fraud detected'
        }

        with self.assertRaises(BankFraudDetectedError):
            self.bank_of_america.make_transaction('1234567890', 100.00, '0987654321')

    @patch('requests.get')
    def test_predict_fraud_success(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = [
            {
                'transaction_id':'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z',
                'fraud': False
            },
            {
                'transaction_id': 'hijklmnop67890',
                'amount': 200.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T15:30:00Z',
                'fraud': True
            }
        ]

        transactions = self.bank_of_america.predict_fraud([
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            },
            {
                'transaction_id': 'hijklmnop67890',
                'amount': 200.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T15:30:00Z'
            }
        ])

        self.assertFalse(transactions[0])
        self.assertTrue(transactions[1])

    @patch('requests.get')
    def test_predict_fraud_failure(self, mock_get):
        mock_get.return_value.status_code = 500
        mock_get.return_value.json.return_value = {
            'error': 'Internal server error'
        }

        with self.assertRaises(BankAPIError):
            self.bank_of_america.predict_fraud([
                {
                    'transaction_id': 'abcdefg12345',
                    'amount': 100.00,
                    'recipient': '0987654321',
                    'timestamp': '2023-03-22T14:30:00Z'
                }
            ])

    @patch('requests.get')
    def test_predict_fraud_invalid_data(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = [
            {
                'transaction_id': 'abcdefg12345',
                'amount': '100.00',
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z',
                'fraud': False
            }
        ]

        with self.assertRaises(BankInvalidDataError):
            self.bank_of_america.predict_fraud([
                {
                    'transaction_id': 'abcdefg12345',
                    'amount': '100.00',
                    'recipient': '0987654321',
                    'timestamp': '2023-03-22T14:30:00Z'
                }
            ])

    @patch('requests.get')
    def test_predict_fraud_rate_limit_exceeded(self, mock_get):
        mock_get.return_value.status_code = 429
        mock_get.return_value.json.return_value = {
            'error': 'Rate limit exceeded'
        }

        with self.assertRaises(BankRateLimitExceededError):
            self.bank_of_america.predict_fraud([
                {
                    'transaction_id': 'abcdefg12345',
                    'amount': 100.00,
                    'recipient':'0987654321',
                    'timestamp': '2023-03-22T14:30:00Z'
                }
            ])

    @patch('requests.get')
    def test_predict_fraud_service_unavailable(self, mock_get):
        mock_get.return_value.status_code = 503
        mock_get.return_value.json.return_value = {
            'error': 'Service unavailable'
        }

        with self.assertRaises(BankServiceUnavailableError):
            self.bank_of_america.predict_fraud([
                {
                    'transaction_id': 'abcdefg12345',
                    'amount': 100.00,
                    'recipient': '0987654321',
                    'timestamp': '2023-03-22T14:30:00Z'
                }
            ])

    @patch('requests.get')
    def test_predict_fraud_maintenance(self, mock_get):
        mock_get.return_value.status_code = 500
        mock_get.return_value.json.return_value = {
            'error': 'Maintenance in progress'
        }

        with self.assertRaises(BankMaintenanceError):
            self.bank_of_america.predict_fraud([
                {
                    'transaction_id': 'abcdefg12345',
                    'amount': 100.00,
                    'recipient': '0987654321',
                    'timestamp': '2023-03-22T14:30:00Z'
                }
            ])

    @patch('requests.get')
    def test_predict_fraud_unknown_error(self, mock_get):
        mock_get.return_value.status_code = 500
        mock_get.return_value.json.return_value = {
            'error': 'Unknown error'
        }

        with self.assertRaises(BankUnknownError):
            self.bank_of_america.predict_fraud([
                {
                    'transaction_id': 'abcdefg12345',
                    'amount': 100.00,
                    'recipient': '0987654321',
                    'timestamp': '2023-03-22T14:30:00Z'
                }
            ])

    @patch('requests.get')
    def test_predict_fraud_authentication_error(self, mock_get):
        mock_get.return_value.status_code = 401
        mock_get.return_value.json.return_value = {
            'error': 'Unauthorized'
        }

        with self.assertRaises(BankAuthenticationError):
            self.bank_of_america.predict_fraud([
                {
                    'transaction_id': 'abcdefg12345',
                    'amount': 100.00,
                    'recipient': '0987654321',
                    'timestamp': '2023-03-22T14:30:00Z'
                }
            ])

    @patch('requests.get')
    def test_predict_fraud_api_error(self, mock_get):
        mock_get.return_value.status_code = 400
        mock_get.return_value.json.return_value = {
            'error': 'Bad Request'
        }

        with self.assertRaises(BankAPIError):
            self.bank_of_america.predict_fraud([
                {
                    'transaction_id': 'abcdefg12345',
                    'amount': 100.00,
                    'recipient': '0987654321',
                    'timestamp': '2023-03-22T14:30:00Z'
                }
            ])

    @patch('banks.bank_of_america.BankOfAmerica._get_api_url')
    def test_predict_fraud_custom_url(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        transactions = self.bank_of_america.predict_fraud([
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            }
        ])

        self.assertEqual(mock_get.call_args[0][0], 'https://custom-api-url.com/predict_fraud')
        self.assertEqual(mock_get.call_args[1]['headers'], {'Authorization': 'Bearer test_api_key'})
        self.assertEqual(mock_get.call_args[1]['json'], [
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            }
        ])

    @patch('banks.bank_of_america.BankOfAmerica._get_api_url')
    def test_predict_fraud_custom_headers(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        custom_headers = {
            'X-Custom-Header': 'value'
        }

        transactions = self.bank_of_america.predict_fraud(
            [
                {
                    'transaction_id': 'abcdefg12345',
                    'amount': 100.00,
                    'recipient': '0987654321',
                    'timestamp': '2023-03-22T14:30:00Z'
                }
            ],
            headers=custom_headers
        )

        self.assertEqual(mock_get.call_args[0][0], 'https://custom-api-url.com/predict_fraud')
        self.assertEqual(mock_get.call_args[1]['headers'], {
            'Authorization': 'Bearer test_api_key',
            'X-Custom-Header': 'value'
        })
        self.assertEqual(mock_get.call_args[1]['json'], [
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            }
        ])

    @patch('banks.bank_of_america.BankOfAmerica._get_api_url')
    def test_predict_fraud_custom_data(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        custom_data = [
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z',
                'custom_field': 'value'
            }
        ]

        transactions = self.bank_of_america.predict_fraud(custom_data)

        self.assertEqual(mock_get.call_args[0][0], 'https://custom-api-url.com/predict_fraud')
        self.assertEqual(mock_get.call_args[1]['headers'], {'Authorization': 'Bearer test_api_key'})
        self.assertEqual(mock_get.call_args[1]['json'], custom_data)

    @patch('banks.bank_of_america.BankOfAmerica._get_api_url')
    def test_predict_fraud_invalid_response(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        mock_get.return_value.status_code = 400
        mock_get.return_value.json.return_value = {
            'error': 'Invalid request'
        }

        with self.assertRaises(InvalidRequestError):
            self.bank_of_america.predict_fraud([
                {
                    'transaction_id': 'abcdefg12345',
                    'amount': 100.00,
                    'recipient': '0987654321',
                    'timestamp': '2023-03-22T14:30:00Z'
                }
            ])

    @patch('banks.bank_of_america.BankOfAmerica._get_api_url')
    def test_predict_fraud_api_error(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        mock_get.return_value.status_code = 500
        mock_get.return_value.json.return_value = {
            'error': 'API error'
        }

        with self.assertRaises(APIError):
            self.bank_of_america.predict_fraud([
                {
                    'transaction_id': 'abcdefg12345',
                    'amount': 100.00,
                    'recipient': '0987654321',
                    'timestamp': '2023-03-22T14:30:00Z'
                }
            ])

    @patch('banks.bank_of_america.BankOfAmerica._get_api_url')
    def test_predict_fraud_rate_limit_error(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        mock_get.return_value.status_code = 429
        mock_get.return_value.json.return_value = {
            'error': 'Rate limit exceeded'
        }

        with self.assertRaises(RateLimitError):
            self.bank_of_america.predict_fraud([
                {
                    'transaction_id': 'abcdefg12345',
                    'amount': 100.00,
                    'recipient': '0987654321',
                    'timestamp': '2023-03-22T14:30:00Z'
                }
            ])

    @patch('banks.bank_of_america.BankOfAmerica._get_api_url')
    def test_predict_fraud_no_error_response(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        mock_get.return_value.status_code = 400
        mock_get.return_value.json.return_value = {
            'transactions': [
                {
                    'transaction_id': 'abcdefg12345',
                    'fraud_score': 0.8
                }
            ]
        }

       
