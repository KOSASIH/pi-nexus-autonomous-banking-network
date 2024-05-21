import unittest
from bank_integration.banks import AbsaBank, CapitecBank, DiscoveryBank, FNBBank, InvestecBank, NedbankBank, StandardBank, CSBBank

class TestBankIntegrations(unittest.TestCase):

    def setUp(self):
        self.absa_bank = AbsaBank('absa_api_key', 'absa_api_secret')
        self.capitec_bank = CapitecBank('capitec_api_key', 'capitec_api_secret')
        self.discovery_bank = DiscoveryBank('discovery_api_key', 'discovery_api_secret')
        self.fnb_bank = FNBBank('fnb_api_key', 'fnb_api_secret')
        self.investec_bank = InvestecBank('investec_api_key', 'investec_api_secret')
        self.nedbank_bank = NedbankBank('nedbank_api_key', 'nedbank_api_secret')
        self.standard_bank = StandardBank('standard_api_key', 'standard_api_secret')
        self.csb_bank = CSBBank('csb_api_key', 'csb_api_secret')

    def test_absa_bank_integration(self):
        account_balance = self.absa_bank.get_account_balance('123456789')
        self.assertGreater(account_balance, 0)

    def test_capitec_bank_integration(self):
        account_balance = self.capitec_bank.get_account_balance('987654321')
        self.assertGreater(account_balance, 0)

    def test_discovery_bank_integration(self):
        account_balance = self.discovery_bank.get_account_balance('111111111')
        self.assertGreater(account_balance, 0)

    def test_fnb_bank_integration(self):
        account_balance = self.fnb_bank.get_account_balance('222222222')
        self.assertGreater(account_balance, 0)

    def test_investec_bank_integration(self):
        account_balance = self.investec_bank.get_account_balance('333333333')
        self.assertGreater(account_balance, 0)

    def test_nedbank_bank_integration(self):
        account_balance = self.nedbank_bank.get_account_balance('444444444')
        self.assertGreater(account_balance, 0)

    def test_standard_bank_integration(self):
        account_balance = self.standard_bank.get_account_balance('555555555')
        self.assertGreater(account_balance, 0)

    def test_csb_bank_integration(self):
        account_balance = self.csb_bank.get_account_balance('666666666')
        self.assertGreater(account_balance, 0)

if __name__ == '__main__':
    unittest.main()
