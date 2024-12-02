// LoanContract.sol
pragma solidity ^0.8.0;

contract LoanContract {
    address public lender;
    address public borrower;
    uint public loanAmount;
    uint public interestRate;
    uint public dueDate;

    constructor(address _borrower, uint _loanAmount, uint _interestRate, uint _dueDate) {
        lender = msg.sender;
        borrower = _borrower;
        loanAmount = _loanAmount;
        interestRate = _interestRate;
        dueDate = _dueDate;
    }

    function repayLoan() public payable {
        require(msg.sender == borrower, "Only borrower can repay the loan");
        require(msg.value == loanAmount + (loanAmount * interestRate / 100), "Incorrect repayment amount");
        // Logic to transfer funds to lender
    }
}
