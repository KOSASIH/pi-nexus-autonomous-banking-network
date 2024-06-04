pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/SafeERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract DeFiContract is Ownable {
    using SafeERC20 for IERC20;

   struct Loan {
        address borrower;
        IERC20 token;
        uint256 amount;
        uint256 interestRate;
        uint256 repaymentPeriod;
        uint256 createdAt;
    }

    mapping(address => Loan[]) public loans;

    event LoanCreated(address indexed borrower, IERC20 token, uint256 amount, uint256 interestRate, uint256 repaymentPeriod, uint256 createdAt);
    event LoanRepaid(address indexed borrower, IERC20 token, uint256 amount, uint256 interestRate, uint256 repaymentPeriod, uint256 repaidAt);

    function createLoan(IERC20 token, uint256 amount, uint256 interestRate, uint256 repaymentPeriod) external {
        //...
    }

    function repayLoan(IERC20 token, uint256 amount) external {
        //...
    }

    function getLoan(address borrower, uint256 index) public view returns (Loan memory) {
        //...
    }
}
