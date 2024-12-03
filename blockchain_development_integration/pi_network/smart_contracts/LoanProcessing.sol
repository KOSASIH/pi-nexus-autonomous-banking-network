// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/math/SafeMath.sol";

contract LoanProcessing is Ownable {
    using SafeMath for uint256;

    enum LoanStatus { Pending, Approved, Rejected, Repaid }

    struct Loan {
        uint256 id;
        address borrower;
        uint256 amount;
        uint256 interestRate; // in basis points (e.g., 500 = 5%)
        uint256 duration; // in seconds
        uint256 startTime;
        LoanStatus status;
    }

    uint256 public loanCounter;
    mapping(uint256 => Loan) public loans;

    // Events
    event LoanApplied(uint256 indexed loanId, address indexed borrower, uint256 amount, uint256 interestRate, uint256 duration);
    event LoanApproved(uint256 indexed loanId);
    event LoanRejected(uint256 indexed loanId);
    event LoanRepaid(uint256 indexed loanId);

    // Apply for a loan
    function applyForLoan(uint256 amount, uint256 interestRate, uint256 duration) external {
        require(amount > 0, "Loan amount must be greater than zero");
        require(interestRate > 0, "Interest rate must be greater than zero");
        require(duration > 0, "Duration must be greater than zero");

        loanCounter++;
        loans[loanCounter] = Loan({
            id: loanCounter,
            borrower: msg.sender,
            amount: amount,
            interestRate: interestRate,
            duration: duration,
            startTime: 0,
            status: LoanStatus.Pending
        });

        emit LoanApplied(loanCounter, msg.sender, amount, interestRate, duration);
    }

    // Approve a loan
    function approveLoan(uint256 loanId) external onlyOwner {
        Loan storage loan = loans[loanId];
        require(loan.status == LoanStatus.Pending, "Loan is not pending");
        
        loan.status = LoanStatus.Approved;
        loan.startTime = block.timestamp;

        emit LoanApproved(loanId);
    }

    // Reject a loan
    function rejectLoan(uint256 loanId) external onlyOwner {
        Loan storage loan = loans[loanId];
        require(loan.status == LoanStatus.Pending, "Loan is not pending");

        loan.status = LoanStatus.Rejected;

        emit LoanRejected(loanId);
    }

    // Repay a loan
    function repayLoan(uint256 loanId) external payable {
        Loan storage loan = loans[loanId];
        require(loan.status == LoanStatus.Approved, "Loan is not approved");
        require(msg.sender == loan.borrower, "Only the borrower can repay the loan");

        uint256 totalRepayment = calculateTotalRepayment(loan.amount, loan.interestRate, loan.duration);
        require(msg.value >= totalRepayment, "Insufficient funds for repayment");

        loan.status = LoanStatus.Repaid;

        emit LoanRepaid(loanId);
    }

    // Calculate total repayment amount
    function calculateTotalRepayment(uint256 amount, uint256 interestRate, uint256 duration) public pure returns (uint256) {
        uint256 interest = amount.mul(interestRate).mul(duration).div(365 days).div(10000); // Interest calculation
        return amount.add(interest);
    }

    // Get loan details
    function getLoanDetails(uint256 loanId) external view returns (Loan memory) {
        return loans[loanId];
    }
}
