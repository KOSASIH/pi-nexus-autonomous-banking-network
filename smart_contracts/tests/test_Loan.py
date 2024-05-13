import pytest
from brownie import Loan

def test_create_loan():
    loan = Loan.deploy({'from': accounts[0]})
    initial_balance = accounts[1].balance()
    loan.createLoan(accounts[1], 100, 10, 120, {'from': accounts[0]})
    final_balance = accounts[1].balance()
    assert final_balance == initial_balance + 100
    assert loan.loans(0).borrower == accounts[1]
    assert loan.loans(0).amount == 100
    assert loan.loans(0).interestRate == 10
    assert loan.loans(0).repaymentTerm == 120

def test_repay_loan():
    loan = Loan.deploy({'from': accounts[0]})
    loan.createLoan(accounts[1], 100, 10, 120, {'from': accounts[0]})
    initial_balance = loan.loans(0).repaymentAmount()
    loan.repayLoan(0, {'from': accounts[1]})
    final_balance = loan.loans(0).repaymentAmount()
    assert final_balance > initial_balance

def test_only_owner():
    loan = Loan.deploy({'from': accounts[0]})
    with pytest.raises(Exception):
        loan.createLoan(accounts[1], 100, 10, 120, {'from': accounts[1]})
