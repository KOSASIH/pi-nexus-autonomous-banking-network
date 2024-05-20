# tests/test_bank_integration.py

import unittest
from unittest.mock import patch, MagicMock

from banks.bank_integration import BankIntegration
from exceptions import BankIntegrationError, BankAuthenticationError, BankAPIError, BankAccountNotFoundError

class TestBankIntegration(unittest.TestCase):
    def setUp(self):
        self.bank_integration = BankIntegration('test_api_key', 'test_api_secret')

    @patch('requests.get')
    def test_get_account_info_success(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            'account_number': '1234567890',
            'account_holder': 'John Doe',
            'bank_type': 'RETAIL',
            'balance': 1000.00
        }

        account_info = self.bank_integration.get_account_info('1234567890')

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
            self.bank_integration.get_account_info('1234567890')

    @patch('requests.post')
    def test_make_transaction_success(self, mock_post):
        mock_post.return_value.status_code = 201
        mock_post.return_value.json.return_value = {
            'transaction_id': 'abcdefg12345'
        }

        transaction_id = self.bank_integration.make_transaction('1234567890', 100.00, '0987654321')

        self.assertEqual(transaction_id, 'abcdefg12345')

    @patch('requests.post')
    def test_make_transaction_failure(self, mock_post):
        mock_post.return_value.status_code = 400
        mock_post.return_value.json.return_value = {
            'error': 'Invalid transaction'
        }

        with self.assertRaises(BankTransactionError):
            self.bank_integration.make_transaction('1234567890', 100.00, '0987654321')

    @patch('requests.post')
    def test_make_transaction_fraud_detected(self, mock_post):
        mock_post.return_value.status_code = 403
        mock_post.return_value.json.return_value = {
            'error': 'Fraud detected'
        }

        with self.assertRaises(BankFraudDetectedError):
            self.bank_integration.make_transaction('1234567890', 100.00, '0987654321')

    @patch('requests.get')
    def test_predict_fraud_success(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = [
            {
                'transaction_id': 'abcdefg12345',
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

        transactions = self.bank_integration.predict_fraud([
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
            self.bank_integration.predict_fraud([
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
            self.bank_integration.predict_fraud([
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
            self.bank_integration.predict_fraud([
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
            self.bank_integration.predict_fraud([
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
            self.bank_integration.predict_fraud([
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
            self.bank_integration.predict_fraud([
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
            self.bank_integration.predict_fraud([
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
            self.bank_integration.predict_fraud([
                {
                    'transaction_id': 'abcdefg12345',
                    'amount': 100.00,
                    'recipient': '0987654321',
                    'timestamp': '2023-03-22T14:30:00Z'
                }
            ])

    @patch('banks.bank_integration.BankIntegration._get_api_url')
    def test_predict_fraud_custom_url(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        transactions = self.bank_integration.predict_fraud([
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

    @patch('banks.bank_integration.BankIntegration._get_api_url')
    def test_predict_fraud_custom_headers(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        custom_headers = {
            'X-Custom-Header': 'value'
        }

        transactions = self.bank_integration.predict_fraud(
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

    @patch('banks.bank_integration.BankIntegration._get_api_url')
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

        transactions = self.bank_integration.predict_fraud(custom_data)

        self.assertEqual(mock_get.call_args[0][0], 'https://custom-api-url.com/predict_fraud')
        self.assertEqual(mock_get.call_args[1]['headers'], {'Authorization': 'Bearer test_api_key'})
        self.assertEqual(mock_get.call_args[1]['json'], custom_data)

    @patch('banks.bank_integration.BankIntegration._get_api_url')
    def test_predict_fraud_custom_timeout(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        transactions = self.bank_integration.predict_fraud([
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            }
        ], timeout=5)

        self.assertEqual(mock_get.call_args[1]['timeout'], 5)

    @patch('banks.bank_integration.BankIntegration._get_api_url')
    def test_predict_fraud_custom_verify(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        transactions = self.bank_integration.predict_fraud([
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            }
        ], verify=False)

        self.assertEqual(mock_get.call_args[1]['verify'], False)

    @patch('banks.bank_integration.BankIntegration._get_api_url')
    def test_predict_fraud_custom_auth(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        custom_auth = ('custom_username', 'custom_password')

        transactions = self.bank_integration.predict_fraud([
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            }
        ], auth=custom_auth)

        self.assertEqual(mock_get.call_args[1]['auth'], custom_auth)

    @patch('banks.bank_integration.BankIntegration._get_api_url')
    def test_predict_fraud_custom_params(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        custom_params = {
            'param1': 'value1',
            'param2': 'value2'
        }

        transactions = self.bank_integration.predict_fraud([
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            }
        ], params=custom_params)

        self.assertEqual(mock_get.call_args[1]['params'], custom_params)

    @patch('banks.bank_integration.BankIntegration._get_api_url')
    def test_predict_fraud_custom_proxies(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        custom_proxies= {
            'http': 'http://10.10.1.10:8080',
            'https': 'http://10.10.1.10:8080'
        }

        transactions = self.bank_integration.predict_fraud([
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            }
        ], proxies=custom_proxies)

        self.assertEqual(mock_get.call_args[1]['proxies'], custom_proxies)

    @patch('banks.bank_integration.BankIntegration._get_api_url')
    def test_predict_fraud_custom_stream(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        transactions = self.bank_integration.predict_fraud([
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            }
        ], stream=True)

        self.assertEqual(mock_get.call_args[1]['stream'], True)

    @patch('banks.bank_integration.BankIntegration._get_api_url')
    def test_predict_fraud_custom_cert(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        custom_cert = ('/path/to/cert.pem', '/path/to/key.pem')

        transactions = self.bank_integration.predict_fraud([
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            }
        ], cert=custom_cert)

        self.assertEqual(mock_get.call_args[1]['cert'], custom_cert)

    @patch('banks.bank_integration.BankIntegration._get_api_url')
    def test_predict_fraud_custom_ssl_context(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        custom_ssl_context = ssl.create_default_context()

        transactions = self.bank_integration.predict_fraud([
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            }
        ], ssl_context=custom_ssl_context)

        self.assertEqual(mock_get.call_args[1]['ssl_context'], custom_ssl_context)

    @patch('banks.bank_integration.BankIntegration._get_api_url')
    def test_predict_fraud_custom_hooks(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        custom_hooks = {
            'response': lambda r: r,
            'request': lambda r: r
        }

        transactions = self.bank_integration.predict_fraud([
            {
                'transaction_id': 'abcdefg12345',
'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            }
        ], hooks=custom_hooks)

        self.assertEqual(mock_get.call_args[1]['hooks'], custom_hooks)

    @patch('banks.bank_integration.BankIntegration._get_api_url')
    def test_predict_fraud_custom_auth_basic(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        custom_auth = ('custom_username', 'custom_password')

        transactions = self.bank_integration.predict_fraud([
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            }
        ], auth=custom_auth, auth_basic=True)

        self.assertEqual(mock_get.call_args[1]['auth'], HTTPBasicAuth(*custom_auth))

    @patch('banks.bank_integration.BankIntegration._get_api_url')
    def test_predict_fraud_custom_auth_digest(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        custom_auth = ('custom_username', 'custom_password')

        transactions = self.bank_integration.predict_fraud([
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            }
        ], auth=custom_auth, auth_digest=True)

        self.assertEqual(mock_get.call_args[1]['auth'], HTTPDigestAuth(*custom_auth))

    @patch('banks.bank_integration.BankIntegration._get_api_url')
    def test_predict_fraud_custom_auth_ntlm(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        custom_auth = ('custom_username', 'custom_password')

        transactions = self.bank_integration.predict_fraud([
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            }
        ], auth=custom_auth, auth_ntlm=True)

        self.assertEqual(mock_get.call_args[1]['auth'], NTLMAuth(*custom_auth))

    @patch('banks.bank_integration.BankIntegration._get_api_url')
    def test_predict_fraud_custom_auth_oauth1(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        custom_auth = ('custom_username', 'custom_password')

        transactions = self.bank_integration.predict_fraud([
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            }
        ], auth=custom_auth, auth_oauth1=True)

        self.assertEqual(mock_get.call_args[1]['auth'], OAuth1(*custom_auth))

    @patch('banks.bank_integration.BankIntegration._get_api_url')
    def test_predict_fraud_custom_auth_oauth2(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        custom_auth = ('custom_username', 'custom_password')

        transactions = self.bank_integration.predict_fraud([
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            }
        ], auth=custom_auth, auth_oauth2=True)

        self.assertEqual(mock_get.call_args[1]['auth'], OAuth2(*custom_auth))

    @patch('banks.bank_integration.BankIntegration._get_api_url')
    def test_predict_fraud_custom_auth_aws4(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        custom_auth = ('custom_username', 'custom_password')

        transactions = self.bank_integration.predict_fraud([
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            }
        ], auth=custom_auth, auth_aws4=True)

        self.assertEqual(mock_get.call_args[1]['auth'], AWS4Auth(*custom_auth))

    @patch('banks.bank_integration.BankIntegration._get_api_url')
    def test_predict_fraud_custom_auth_aws_sigv4(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        custom_auth = ('custom_username', 'custom_password')

        transactions = self.bank_integration.predict_fraud([
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            }
        ], auth=custom_auth, auth_aws_sigv4=True)

        self.assertEqual(mock_get.call_args[1]['auth'], AWS4Auth(*custom_auth))

    @patch('banks.bank_integration.BankIntegration._get_api_url')
    def test_predict_fraud_custom_auth_openid(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        custom_auth = ('custom_username', 'custom_password')

        transactions = self.bank_integration.predict_fraud([
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            }
        ], auth=custom_auth, auth_openid=True)

        self.assertEqual(mock_get.call_args[1]['auth'], OpenIDAuth(*custom_auth))

    @patch('banks.bank_integration.BankIntegration._get_api_url')
    def test_predict_fraud_custom_auth_kerberos(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        custom_auth = ('custom_username', 'custom_password')

        transactions = self.bank_integration.predict_fraud([
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            }
        ], auth=custom_auth, auth_kerberos=True)

        self.assertEqual(mock_get.call_args[1]['auth'], KerberosAuth(*custom_auth))

    @patch('banks.bank_integration.BankIntegration._get_api_url')
    def test_predict_fraud_custom_auth_ntlm_win32(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        custom_auth = ('custom_username', 'custom_password')

        transactions = self.bank_integration.predict_fraud([
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            }
        ], auth=custom_auth, auth_ntlm_win32=True)

        self.assertEqual(mock_get.call_args[1]['auth'], NTLMWin32Auth(*custom_auth))

    @patch('banks.bank_integration.BankIntegration._get_api_url')
    def test_predict_fraud_custom_auth_http_proxy(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        custom_auth = ('custom_username', 'custom_password')

        transactions = self.bank_integration.predict_fraud([
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            }
        ], auth=custom_auth, auth_http_proxy=True)

        self.assertEqual(mock_get.call_args[1]['auth'], HTTPProxyAuth(*custom_auth))

    @patch('banks.bank_integration.BankIntegration._get_api_url')
    def test_predict_fraud_custom_auth_http_basic_auth(self, mock_get_api_url):
        mock_get_api_url.return_value = 'https://custom-api-url.com'

        custom_auth = ('custom_username', 'custom_password')

        transactions = self.bank_integration.predict_fraud([
            {
                'transaction_id': 'abcdefg12345',
                'amount': 100.00,
                'recipient': '0987654321',
                'timestamp': '2023-03-22T14:30:00Z'
            }
        ], auth=custom_auth, auth_http_basic_auth=True)

        self.assertEqual(mock_get.call_args[1]['auth'], HTTPBasicAuth(*custom_auth))

    @patch('banks.bank_integration.BankIntegration._get_api_url')
    def test_predict_fraud_custom_auth_http_digest_auth(self, mock_get
