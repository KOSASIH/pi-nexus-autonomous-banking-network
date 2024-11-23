// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract LendingProtocol is Ownable {
    struct Loan {
        address borrower;
        uint256 amount;
        uint256 interestRate; // in basis points
        uint256 duration; // in seconds
        uint256 startTime;
        bool isActive;
    }

    mapping(address => Loan) public loans;
    IERC20 public stablecoin;

    event LoanRequested(address indexed borrower, uint256 amount, uint256 interestRate, uint256 duration);
    event LoanRepaid(address indexed borrower, uint256 amount);

    constructor(IERC20 _stablecoin) {
        stablecoin = _stablecoin;
    }

    // Request a loan
    function requestLoan(uint256 _amount, uint256 _interestRate, uint256 _duration) public {
        require(loans[msg.sender].isActive == false, "Existing loan must be repaid first.");
        require(stablecoin.balanceOf(address(this)) >= _amount, "Insufficient liquidity.");

        loans[msg.sender] = Loan(msg.sender, _amount, _interestRate, _duration, block.timestamp, true);
        stablecoin.transfer(msg.sender, _amount);
        emit LoanRequested(msg.sender, _amount, _interestRate, _duration);
    }

    // Repay the loan
    function repayLoan() public {
        Loan storage loan = loans[msg.sender];
        require(loan.isActive, "No active loan found.");

        uint256 totalRepayment = loan.amount + (loan.amount * loan.interestRate / 10000);
        require(stablecoin.balanceOf(msg.sender) >= totalRepayment, "Insufficient repayment amount.");

        stablecoin.transferFrom(msg.sender, address(this), totalRepayment);
        loan.isActive = false;
        emit LoanRepaid(msg.sender, totalRepayment);
    }

    // Get loan details
    function getLoanDetails(address _borrower) public view returns (uint256, uint256, uint256, uint256, bool) {
        Loan storage loan = loans[_borrower];
        return (loan.amount, loan.interestRate, loan.duration, loan.startTime, loan.isActive);
    }
}
