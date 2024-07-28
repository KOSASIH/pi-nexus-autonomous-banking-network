pragma solidity ^0.8.0;

contract LendingContract {
    address private owner;
    mapping (address => uint256) public creditScores;
    mapping (address => uint256) public loanAmounts;
    mapping (address => uint256) public interestRates;

    constructor() {
        owner = msg.sender;
    }

    function applyForLoan(uint256 _creditScore, uint256 _loanAmount, uint256 _interestRate) public {
        require(_creditScore > 0, "Credit score must be greater than 0");
        require(_loanAmount > 0, "Loan amount must be greater than 0");
        require(_interestRate > 0, "Interest rate must be greater than 0");

        creditScores[msg.sender] = _creditScore;
        loanAmounts[msg.sender] = _loanAmount;
        interestRates[msg.sender] = _interestRate;
    }

    function getLoanStatus(address _borrower) public view returns (uint256, uint256, uint256) {
        return (creditScores[_borrower], loanAmounts[_borrower], interestRates[_borrower]);
    }

    function repayLoan(uint256 _amount) public {
        require(_amount > 0, "Repayment amount must be greater than 0");

        // Calculate interest
        uint256 interest = (_amount * interestRates[msg.sender]) / 100;

        // Update loan amount
        loanAmounts[msg.sender] -= _amount + interest;
    }
}
