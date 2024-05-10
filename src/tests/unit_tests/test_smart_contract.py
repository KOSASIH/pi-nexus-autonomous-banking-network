import unittest
from smart_contract import SmartContract

class TestSmartContract(unittest.TestCase):
    def test_create_smart_contract(self):
        contract = SmartContract('contract_owner', 'contract_code')
        self.assertEqual(contract.owner, 'contract_owner')
        self.assertEqual(contract.code, 'contract_code')
        self.assertEqual(contract.hash, None)

    def test_hash(self):
        contract = SmartContract('contract_owner', 'contract_code')
        contract.hash = None
        contract.mine_hash()
        self.assertNotEqual(contract.hash, None)

    def test_mine_hash(self):
        contract = SmartContract('contract_owner', 'contract_code')
        contract.mine_hash(difficulty=2)
        self.assertEqual(contract.hash[0:2], '00')

    def test_execute_contract(self):
        contract = SmartContract('contract_owner', 'contract_code')
result = contract.execute_contract('input_data')
        self.assertIsInstance(result, str)

if __name__ == '__main__':
    unittest.main()
