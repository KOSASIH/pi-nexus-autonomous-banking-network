from typing import Dict

class DeFiLendingContract:
    def __init__(self):
        self.loans: Dict[str, Dict[str, int]] = {}
        self.lenders: Dict[str, Dict[str, int]] = {}
        self.borrowers: Dict[str, Dict[str, int]] = {}

    def create_loan(self, loan_id: str, amount: int, interest_rate: int, lender_id: str, borrower_id: str):
        # Create a new loan
        self.loans[loan_id] = {
            "amount": amount,
            "interest_rate": interest_rate,
            "lender_id": lender_id,
            "borrower_id": borrower_id
        }
        self.lenders[lender_id] = self.lenders.get(lender_id, {})
        self.lenders[lender_id][loan_id] = amount
        self.borrowers[borrower_id] = self.borrowers.get(borrower_id, {})
        self.borrowers[borrower_id][loan_id] = amount

    def deposit(self, lender_id: str, amount: int):
        # Deposit funds into a lender's account
        self.lenders[lender_id] = self.lenders.get(lender_id, {})
        self.lenders[lender_id]["balance"] = self.lenders[lender_id].get("balance", 0) + amount

    def withdraw(self, lender_id: str, amount: int):
        # Withdraw funds from a lender's account
        if lender_id not in self.lenders or self.lenders[lender_id].get("balance", 0) < amount:
            raise ValueError("Insufficient balance")
        self.lenders[lender_id]["balance"] -= amount

    def borrow(self, borrower_id: str, loan_id: str, amount: int):
        # Borrow funds from a lender
        if loan_id not in self.loans or self.loans[loan_id]["lender_id"] not in self.lenders:
            raise ValueError("Loan not found")
        if self.lenders[self.loans[loan_id]["lender_id"]].get("balance", 0) < amount:
            raise ValueError("Insufficient lender balance")
        self.lenders[self.loans[loan_id]["lender_id"]]["balance"] -= amount
        self.borrowers[borrower_id][loan_id] = self.borrowers[borrower_id].get(loan_id, 0) + amount

    def repay(self, borrower_id: str, loan_id: str, amount: int):
        # Repay a loan
        if loan_id not in self.loans or borrower_id not in self.borrowers:
            raise ValueError("Loan not found")
        if self.borrowers[borrower_id][loan_id] < amount:
            raise ValueError("Insufficient borrower balance")
        self.borrowers[borrower_id][loan_id] -= amount
        self.lenders[self.loans[loan_id]["lender_id"]]["balance"] += amount

    def get_loan_info(self, loan_id: str):
        # Get the information of a loan
        return self.loans.get(loan_id, {})

    def get_lender_info(self, lender_id: str):
        # Get the information of a lender
        return self.lenders.get(lender_id, {})

    def get_borrower_info(self, borrower_id: str):
        # Get the information of a borrower
        return self.borrowers.get(borrower_id, {})
