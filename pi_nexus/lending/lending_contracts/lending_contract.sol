pragma solidity ^0.8.0;

interface Token {
    function transferFrom(address from, address to, uint256 value) external returns (bool success);
}

contract LendingContract {
    struct Loan {
        address user;
        uint256 amount;
        uint256 interestRate;
        uint256 maturityDate;
        bool repaid;
    }

    mapping(address => mapping(uint256 => Loan)) public loans;

    event LoanCreated(address indexed user, uint256 indexed loanId, uint256 amount, uint256 interestRate, uint256 maturityDate);

    function createLoan(address _token, uint256 _amount, uint256 _interestRate, uint256 _maturityDate) public {
        Loan memory newLoan;
        newLoan.user = msg.sender;
        newLoan.amount = _amount;
        newLoan.interestRate = _interestRate;
        newLoan.maturityDate = _maturityDate;
        newLoan.repaid = false;

        uint256 loanId = loans[msg.sender].length;
        loans[msg.sender][loanId] = newLoan;

        emit LoanCreated(msg.sender, loanId, _amount, _interestRate, _maturityDate);
    }

    function repayLoan(address _token, uint256 _loanId, uint256 _amount) public {
        Loan storage loan = loans[msg.sender][_loanId];
        require(!loan.repaid, "This loan has already been repaid.");

        Token(_token).transferFrom(address(this), msg.sender, _amount);

        loan.repaid = true;
    }

    function getLoan(address _user, uint256 _loanId) public view returns (address, uint256, uint256, uint256, bool) {
        Loan storage loan = loans[_user][_loanId];
        return (loan.user, loan.amount, loan.interestRate, loan.maturityDate, loan.repaid);
    }
}
