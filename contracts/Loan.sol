pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

contract Loan is Ownable, Pausable {
    // Loan variables
    struct LoanStruct {
        address borrower;
        uint256 amount;
        uint256 interestRate;
        uint256 repaymentTerm;
        uint256 repaymentAmount;
        bool isRepaid;
    }

    LoanStruct[] public loans;

    // Event for when a loan is created
    event LoanCreated(address indexed borrower, uint256 indexed loanId, uint256 amount, uint256 interestRate, uint256 repaymentTerm, uint256 repaymentAmount);

    // Event for when a loan is repaid
    event LoanRepaid(address indexed borrower, uint256 indexed loanId);

    // Function to create a loan
    function createLoan(address borrower, uint256 amount, uint256 interestRate, uint256 repaymentTerm, uint256 repaymentAmount) public onlyOwner {
        require(!paused(), "Contract is paused");

        // Create a new loan with the given parameters
        LoanStruct memory newLoan = LoanStruct({
            borrower: borrower,
            amount: amount,
            interestRate: interestRate,
            repaymentTerm: repaymentTerm,
            repaymentAmount: repaymentAmount,
            isRepaid: false
        });

        // Add the new loan to the loans array
        loans.push(newLoan);

        // Emit an event for when a loan is created
        emit LoanCreated(borrower, loans.length - 1, amount, interestRate, repaymentTerm, repaymentAmount);
    }

    // Function to repay a loan
    function repayLoan(uint256 loanId) public onlyOwner {
        require(!paused(), "Contract is paused");

        // Get the loan with the given loanId
        LoanStruct storage loan = loans[loanId];

        // Check if the loan has already been repaid
        require(!loan.isRepaid, "Loan has already been repaid");

        // Calculate the total amount to be repaid, including interest
        uint256 totalRepaymentAmount = loan.amount + (loan.amount * loan.interestRate / 10000);

        // Check if the repayment amount is sufficient
        require(totalRepaymentAmount <= loan.repaymentAmount, "Insufficient repayment amount");

        // Set the loan as repaid
        loan.isRepaid = true;

        // Emit an event for when a loan is repaid
        emit LoanRepaid(loan.borrower, loanId);
    }

    // Function to pause the contract
    function pause() public onlyOwner {
        _pause();
    }

    // Function to unpause the contract
    function unpause() public onlyOwner {
        _unpause();
    }
}
