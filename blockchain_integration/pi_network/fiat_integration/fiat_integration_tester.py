import unittest
from fiat_gateway_manager import FiatGatewayManager
from fiat_currency_converter import FiatCurrencyConverter
from transaction_history_manager import TransactionHistoryManager
from fiat_wallet_manager import FiatWalletManager
from compliance_manager import ComplianceManager
from risk_management import RiskManagement
from fiat_payment_processor import FiatPaymentProcessor
from exchange_rate_manager import ExchangeRateManager

class FiatIntegrationTester(unittest.TestCase):
    def setUp(self):
        self.fiat_gateway_manager = FiatGatewayManager()
        self.fiat_currency_converter = FiatCurrencyConverter()
        self.transaction_history_manager = TransactionHistoryManager()
        self.fiat_wallet_manager = FiatWalletManager()
        self.compliance_manager = ComplianceManager()
        self.risk_management = RiskManagement()
        self.fiat_payment_processor = FiatPaymentProcessor()
        self.exchange_rate_manager = ExchangeRateManager()

    def test_fiat_transaction(self):
        # Test fiat transaction
        fiat_transaction = self.fiat_payment_processor.process_fiat_transaction("USD", 100)
        self.assertIsNotNone(fiat_transaction)

    def test_error_handling(self):
        # Test error handling
        try:
            self.fiat_payment_processor.process_fiat_transaction("Invalid Currency", 100)
            self.fail("Expected ValueError")
        except ValueError:
            pass

    def test_compliance(self):
        # Test compliance
        compliance_result = self.compliance_manager.check_compliance("USD", 100)
        self.assertTrue(compliance_result)

    def test_risk_management(self):
        # Test risk management
        risk_result = self.risk_management.assess_risk("USD", 100)
        self.assertIsNotNone(risk_result)

    def test_exchange_rate(self):
        # Test exchange rate
        exchange_rate = self.exchange_rate_manager.get_exchange_rate("USD", "EUR")
        self.assertIsNotNone(exchange_rate)

    def test_fiat_wallet(self):
        # Test fiat wallet
        fiat_wallet = self.fiat_wallet_manager.create_fiat_wallet("USD", 100)
        self.assertIsNotNone(fiat_wallet)

    def test_transaction_history(self):
        # Test transaction history
        transaction_history = self.transaction_history_manager.get_transaction_history("USD")
        self.assertIsNotNone(transaction_history)

if __name__ == "__main__":
    unittest.main()
