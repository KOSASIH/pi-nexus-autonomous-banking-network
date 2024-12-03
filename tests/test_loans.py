import unittest
from loans import LoanManager  # Assuming you have a LoanManager class

class TestLoanManager(unittest.TestCase):
    def setUp(self):
        self.loan_manager = LoanManager()

    def test_create_loan(self):
        loan = self.loan_manager.create_loan("Alice", 1000, 5)
        self.assertIsNotNone(loan)
        self.assertEqual(loan.amount, 1000)

    def test_loan_repayment(self):
        loan = self.loan_manager.create_loan("Bob", 500, 3)
        self.loan_manager.repay_loan(loan.id, 200)
        self.assertEqual(loan.remaining_balance, 300)

    def test_loan_not_found(self):
        with self.assertRaises(ValueError):
            self.loan_manager.repay_loan("nonexistent_id", 100)

if __name__ == "__main__":
    unittest.main()
