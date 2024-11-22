// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract LoanContract {
    struct Loan {
        uint256 amount;
        uint256 interestRate; // in basis points (1/100th of a percent)
        uint256 duration; // in seconds
        uint256 startTime;
        address borrower;
        bool isActive;
    }

    mapping(address => Loan) public loans;
    event LoanRequested(address indexed borrower, uint256 amount, uint256 interestRate, uint256 duration);
    event LoanRepaid(address indexed borrower, uint256 amount);

    // Request a loan
    function requestLoan(uint256 _amount, uint256 _interestRate, uint256 _duration) public {
        require(loans[msg.sender].isActive == false, "Existing loan must be repaid first.");
        loans[msg.sender] = Loan(_amount, _interestRate, _duration, block.timestamp, msg.sender, true);
        emit LoanRequested(msg.sender, _amount, _interestRate, _duration);
    }

    // Repay the loan
    function repayLoan() public payable {
        Loan storage loan = loans[msg.sender];
        require(loan.isActive, "No active loan found.");
        uint256 totalRepayment = loan.amount + (loan.amount * loan.interestRate / 10000);
        require(msg.value >= totalRepayment, "Insufficient repayment amount.");

        loan.isActive = false;
        emit LoanRepaid(msg.sender, totalRepayment);
        payable(address(this)).transfer(msg.value);
    }

    // Get loan details
    function getLoanDetails(address _borrower) public view returns (uint256, uint256, uint256, uint256, bool) {
        Loan storage loan = loans[_borrower];
        return (loan.amount, loan.interestRate, loan.duration, loan.startTime, loan.isActive);
    }
}
