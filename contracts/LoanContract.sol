// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract LoanContract {
    struct Loan {
        uint256 amount;
        uint256 interestRate; // in basis points (1/100th of a percent)
        uint256 term; // in seconds
        uint256 startTime;
        address borrower;
        bool isRepaid;
    }

    mapping(uint256 => Loan) public loans;
    uint256 public loanCounter;

    event LoanCreated(uint256 indexed loanId, address indexed borrower, uint256 amount, uint256 interestRate, uint256 term);
    event LoanRepaid(uint256 indexed loanId, address indexed borrower);
    event LoanDefaulted(uint256 indexed loanId, address indexed borrower);

    modifier onlyBorrower(uint256 loanId) {
        require(msg.sender == loans[loanId].borrower, "Not the borrower");
        _;
    }

    modifier loanExists(uint256 loanId) {
        require(loanId < loanCounter, "Loan does not exist");
        _;
    }

    // Function to create a new loan
    function createLoan(uint256 _amount, uint256 _interestRate, uint256 _term) public {
        require(_amount > 0, "Loan amount must be greater than zero");
        require(_interestRate > 0, "Interest rate must be greater than zero");
        require(_term > 0, "Loan term must be greater than zero");

        loans[loanCounter] = Loan({
            amount: _amount,
            interestRate: _interestRate,
            term: _term,
            startTime: block.timestamp,
            borrower: msg.sender,
            isRepaid: false
        });

        emit LoanCreated(loanCounter, msg.sender, _amount, _interestRate, _term);
        loanCounter++;
    }

    // Function to calculate the total repayment amount
    function calculateRepaymentAmount(uint256 loanId) public view loanExists(loanId) returns (uint256) {
        Loan memory loan = loans[loanId];
        uint256 interest = (loan.amount * loan.interestRate * loan.term) / (365 days * 10000); // Annual interest calculation
        return loan.amount + interest;
    }

    // Function to repay the loan
    function repayLoan(uint256 loanId) public payable onlyBorrower(loanId) loanExists(loanId) {
        require(!loans[loanId].isRepaid, "Loan already repaid");
        uint256 totalRepayment = calculateRepaymentAmount(loanId);
        require(msg.value >= totalRepayment, "Insufficient funds to repay the loan");

        loans[loanId].isRepaid = true;
        emit LoanRepaid(loanId, msg.sender);

        // Transfer excess funds back to the borrower
        if (msg.value > totalRepayment) {
            payable(msg.sender).transfer(msg.value - totalRepayment);
        }
    }

    // Function to check if a loan is in default
    function checkDefault(uint256 loanId) public view loanExists(loanId) returns (bool) {
        Loan memory loan = loans[loanId];
        if (!loan.isRepaid && (block.timestamp > loan.startTime + loan.term)) {
            return true;
        }
        return false;
    }

    // Function to handle loan defaults
    function handleDefault(uint256 loanId) public loanExists(loanId) {
        require(checkDefault(loanId), "Loan is not in default");
        emit LoanDefaulted(loanId, loans[loanId].borrower);
        // Additional logic for handling defaults can be implemented here
    }
}
