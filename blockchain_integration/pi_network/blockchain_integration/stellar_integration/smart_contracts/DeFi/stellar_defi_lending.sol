pragma solidity ^0.8.0;

import "https://github.com/stellar/solidity-stellar/blob/master/contracts/StellarToken.sol";

contract DeFiLending {
    address private owner;
    mapping (address => uint256) public borrowerBalances;
    mapping (address => uint256) public lenderBalances;
    mapping (address => uint256) public loanRequests;
    mapping (address => uint256) public loanOffers;

    constructor() public {
        owner = msg.sender;
    }

    function requestLoan(uint256 _amount, uint256 _interestRate) public {
        require(msg.sender!= owner, "Only users can request loans");
        loanRequests[msg.sender] = _amount;
        emit LoanRequest(msg.sender, _amount, _interestRate);
    }

    function offerLoan(uint256 _amount, uint256 _interestRate) public {
        require(msg.sender!= owner, "Only users can offer loans");
        loanOffers[msg.sender] = _amount;
        emit LoanOffer(msg.sender, _amount, _interestRate);
    }

    function matchLoan(address _borrower, address _lender) public {
        require(msg.sender == owner, "Only the owner can match loans");
        uint256 borrowerAmount = loanRequests[_borrower];
        uint256 lenderAmount = loanOffers[_lender];
        require(borrowerAmount <= lenderAmount, "Insufficient lender balance");
        borrowerBalances[_borrower] += borrowerAmount;
        lenderBalances[_lender] -= borrowerAmount;
        emit LoanMatched(_borrower, _lender, borrowerAmount);
    }

    function repayLoan(uint256 _amount) public {
        require(msg.sender!= owner, "Only users can repay loans");
        uint256 borrowerBalance = borrowerBalances[msg.sender];
        require(_amount <= borrowerBalance, "Insufficient borrower balance");
        borrowerBalances[msg.sender] -= _amount;
        emit LoanRepaid(msg.sender, _amount);
    }
}
