pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";

contract PILoan {
    using SafeMath for uint256;

    address public owner;
    mapping (address => mapping (address => uint256)) public loans;

    uint256 public interestRate = 10; // 10% annual interest rate

    event LoanRequest(address indexed borrower, address indexed lender, uint256 amount, uint256 interest);
    event LoanRepayment(address indexed borrower, address indexed lender, uint256 amount);

    constructor() public {
        owner = msg.sender;
    }

    function requestLoan(address _lender, uint256 _amount) public {
        require(_amount > 0, "Invalid loan amount");
        require(loans[msg.sender][_lender] == 0, "Loan already exists");

        uint256 interest = (_amount * interestRate) / 100;
        loans[msg.sender][_lender] = _amount.add(interest);

        emit LoanRequest(msg.sender, _lender, _amount, interest);
    }

    function repayLoan(address _lender, uint256 _amount) public {
        require(loans[msg.sender][_lender] != 0, "No loan exists");
        require(_amount <= loans[msg.sender][_lender], "Insufficient repayment amount");

        loans[msg.sender][_lender] = loans[msg.sender][_lender].sub(_amount);

        emit LoanRepayment(msg.sender, _lender, _amount);
     }

     function approveLoanRequest(address _borrower, address _lender) public onlyOwner {
         require(loans[_borrower][_lender] > 0, "No loan request exists");

         loans[_borrower][_lender] = 0;

         emit LoanApproval(_borrower, _lender);
      }
}
