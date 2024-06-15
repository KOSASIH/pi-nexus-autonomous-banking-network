pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/security/ReentrancyGuard.sol";

contract SmartContractLoans is ReentrancyGuard {
    using SafeMath for uint256;

    // Mapping of loan requests
    mapping (address => LoanRequest) public loanRequests;

    // Mapping of loan approvals
    mapping (address => bool) public loanApprovals;

    // Event emitted when a new loan request is created
    event NewLoanRequest(address indexed borrower, uint256 indexed amount, uint256 indexed interestRate, uint256 indexed repaymentTerm);

    // Event emitted when a loan request is approved
    event LoanRequestApprove(address indexed borrower, uint256 indexed amount, uint256 indexed interestRate, uint256 indexed repaymentTerm);

    // Event emitted when a loan is repaid
    event LoanRepayment(address indexed borrower, uint256 indexed amount);

    // Function to create a new loan request
    function createLoanRequest(address borrower, uint256 amount, uint256 interestRate, uint256 repaymentTerm) public {
        require(!loanRequests[borrower].exists, "Loan request already exists");
        loanRequests[borrower] = LoanRequest(amount, interestRate, repaymentTerm, false);
        emit NewLoanRequest(borrower, amount, interestRate, repaymentTerm);
    }

    // Function to approve a loan request
    function approveLoanRequest(address borrower) public {
        require(loanRequests[borrower].exists, "Loan request does not exist");
        require(!loanApprovals[borrower], "Loan request already approved");
        loanApprovals[borrower] = true;
        emit LoanRequestApprove(borrower, loanRequests[borrower].amount, loanRequests[borrower].interestRate, loanRequests[borrower].repaymentTerm);
    }

    // Function to repay a loan
    function repayLoan(address borrower, uint256 amount) public {
        require(loanApprovals[borrower], "Loan request not approved");
        require(loanRequests[borrower].amount >= amount, "Insufficient repayment amount");
        loanRequests[borrower].amount = loanRequests[borrower].amount.sub(amount);
        emit LoanRepayment(borrower, amount);
    }
}

struct LoanRequest {
    uint256 amount;
    uint256 interestRate;
    uint256 repaymentTerm;
    bool exists;
}
