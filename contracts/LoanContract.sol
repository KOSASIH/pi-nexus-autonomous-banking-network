// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract LoanContract {
    enum LoanState { Active, Repaid, Defaulted, Canceled }

    struct Loan {
        uint256 amount;
        uint256 interestRate; // in basis points (1/100th of a percent)
        uint256 term; // in seconds
        uint256 startTime;
        address borrower;
        address lender;
        LoanState state;
        uint256 collateralAmount;
        address collateralAsset; // ERC20 token address for collateral
        uint256 lateFee; // Late fee in basis points
        uint256 repaymentInterval; // Interval in seconds for repayments
        uint256 nextPaymentDue; // Timestamp for the next payment due
    }

    mapping(uint256 => Loan) public loans;
    uint256 public loanCounter;

    event LoanCreated(uint256 indexed loanId, address indexed borrower, address indexed lender, uint256 amount, uint256 interestRate, uint256 term);
    event LoanRepaid(uint256 indexed loanId, address indexed borrower, uint256 amount);
    event LoanDefaulted(uint256 indexed loanId, address indexed borrower);
    event LoanCanceled(uint256 indexed loanId, address indexed borrower);
    event CollateralSeized(uint256 indexed loanId, address indexed borrower, uint256 collateralAmount);

    modifier onlyBorrower(uint256 loanId) {
        require(msg.sender == loans[loanId].borrower, "Not the borrower");
        _;
    }

    modifier onlyLender(uint256 loanId) {
        require(msg.sender == loans[loanId].lender, "Not the lender");
        _;
    }

    modifier loanExists(uint256 loanId) {
        require(loanId < loanCounter, "Loan does not exist");
        _;
    }

    modifier inState(uint256 loanId, LoanState state) {
        require(loans[loanId].state == state, "Invalid loan state");
        _;
    }

    // Function to create a new loan
    function createLoan(address _lender, uint256 _amount, uint256 _interestRate, uint256 _term, uint256 _collateralAmount, address _collateralAsset, uint256 _lateFee, uint256 _repaymentInterval) public {
        require(_amount > 0, "Loan amount must be greater than zero");
        require(_interestRate > 0, "Interest rate must be greater than zero");
        require(_term > 0, "Loan term must be greater than zero");
        require(_collateralAmount > 0, "Collateral amount must be greater than zero");

        // Transfer collateral to the contract
        require(IERC20(_collateralAsset).transferFrom(msg.sender, address(this), _collateralAmount), "Collateral transfer failed");

        loans[loanCounter] = Loan({
            amount: _amount,
            interestRate: _interestRate,
            term: _term,
            startTime: block.timestamp,
            borrower: msg.sender,
            lender: _lender,
            state: LoanState.Active,
            collateralAmount: _collateralAmount,
            collateralAsset: _collateralAsset,
            lateFee: _lateFee,
            repaymentInterval: _repaymentInterval,
            nextPaymentDue: block.timestamp + _repaymentInterval
        });

        emit LoanCreated(loanCounter, msg.sender, _lender, _amount, _interestRate, _term);
        loanCounter++;
    }

    // Function to calculate the total repayment amount
    function calculateRepaymentAmount(uint256 loanId) public view loanExists(loanId) returns (uint256) {
        Loan memory loan = loans[loanId];
        uint256 interest = (loan.amount * loan.interestRate * loan.term) / (365 days * 10000); // Annual interest calculation
        return loan.amount + interest;
    }

    // Function to repay the loan
    function repayLoan(uint256 loanId) public payable onlyBorrower(loanId) loanExists(loanId) inState(loanId, LoanState.Active) {
        require(msg.value > 0, "Repayment amount must be greater than zero");
        require(msg.value <= calculateRepaymentAmount(loanId), "Repayment exceeds total amount due");
        require(block .timestamp >= loans[loanId].nextPaymentDue, "Payment is not due yet");

        loans[loanId].nextPaymentDue += loans[loanId].repaymentInterval;

        // Transfer repayment to the lender
        payable(loans[loanId].lender).transfer(msg.value);
        emit LoanRepaid(loanId, msg.sender, msg.value);

        // Check if the loan is fully repaid
        if (msg.value == calculateRepaymentAmount(loanId)) {
            loans[loanId].state = LoanState.Repaid;
        }
    }

    // Function to default the loan
    function defaultLoan(uint256 loanId) public onlyLender(loanId) loanExists(loanId) inState(loanId, LoanState.Active) {
        require(block.timestamp >= loans[loanId].startTime + loans[loanId].term, "Loan term has not expired yet");

        loans[loanId].state = LoanState.Defaulted;

        // Seize collateral
        require(IERC20(loans[loanId].collateralAsset).transfer(loans[loanId].lender, loans[loanId].collateralAmount), "Collateral transfer failed");
        emit CollateralSeized(loanId, loans[loanId].borrower, loans[loanId].collateralAmount);
    }

    // Function to cancel the loan
    function cancelLoan(uint256 loanId) public onlyBorrower(loanId) loanExists(loanId) inState(loanId, LoanState.Active) {
        loans[loanId].state = LoanState.Canceled;

        // Refund collateral
        require(IERC20(loans[loanId].collateralAsset).transfer(loans[loanId].borrower, loans[loanId].collateralAmount), "Collateral refund failed");
        emit LoanCanceled(loanId, msg.sender);
    }

    // Function to get loan details
    function getLoanDetails(uint256 loanId) public view loanExists(loanId) returns (uint256, uint256, uint256, address, address, LoanState, uint256, address, uint256, uint256) {
        Loan memory loan = loans[loanId];
        return (loan.amount, loan.interestRate, loan.term, loan.borrower, loan.lender, loan.state, loan.collateralAmount, loan.collateralAsset, loan.lateFee, loan.repaymentInterval);
    }
}
