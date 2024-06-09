pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Address.sol";

contract PI_Nexus {
    using SafeERC20 for IERC20;
    using SafeMath for uint256;
    using Address for address;

    // Events
    event NewUser(address indexed user);
    event Deposit(address indexed user, uint256 amount);
    event Withdrawal(address indexed user, uint256 amount);
    event Transfer(address indexed from, address indexed to, uint256 amount);
    event LoanRequest(address indexed user, uint256 amount);
    event LoanApproval(address indexed user, uint256 amount);
    event LoanRepayment(address indexed user, uint256 amount);
    event InterestAccrued(address indexed user, uint256 amount);

    // Structs
    struct User {
        address userAddress;
        uint256 balance;
        uint256 loanAmount;
        uint256 interestAccrued;
        bool isLoanActive;
    }

    // Mappings
    mapping (address => User) public users;
    mapping (address => uint256) public userLoanRequests;

    // Constants
    uint256 public constant MIN_DEPOSIT = 0.1 ether;
    uint256 public constant MAX_LOAN_AMOUNT = 1000 ether;
    uint256 public constant INTEREST_RATE = 5; // 5% per annum
    uint256 public constant LOAN_APPROVAL_THRESHOLD = 50; // 50% of total deposits

    // State variables
    address public owner;
    uint256 public totalDeposits;
    uint256 public totalLoans;
    uint256 public totalInterestAccrued;

    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function");
        _;
    }

    modifier onlyUser(address user) {
        require(users[user].userAddress == user, "Only the user can call this function");
        _;
    }

    // Constructor
    constructor() public {
        owner = msg.sender;
    }

    // Functions
    function deposit() public payable {
        require(msg.value >= MIN_DEPOSIT, "Deposit amount is too low");
        users[msg.sender].balance += msg.value;
        totalDeposits += msg.value;
        emit Deposit(msg.sender, msg.value);
    }

    function withdraw(uint256 amount) public {
        require(users[msg.sender].balance >= amount, "Insufficient balance");
        users[msg.sender].balance -= amount;
        totalDeposits -= amount;
        msg.sender.transfer(amount);
        emit Withdrawal(msg.sender, amount);
    }

    function transfer(address to, uint256 amount) public {
        require(users[msg.sender].balance >= amount, "Insufficient balance");
        users[msg.sender].balance -= amount;
        users[to].balance += amount;
        emit Transfer(msg.sender, to, amount);
    }

    function requestLoan(uint256 amount) public {
        require(amount <= MAX_LOAN_AMOUNT, "Loan amount exceeds maximum limit");
        userLoanRequests[msg.sender] = amount;
        emit LoanRequest(msg.sender, amount);
    }

    function approveLoan(address user) public onlyOwner {
        require(userLoanRequests[user] > 0, "No loan request found");
        uint256 loanAmount = userLoanRequests[user];
        users[user].loanAmount += loanAmount;
        users[user].isLoanActive = true;
        totalLoans += loanAmount;
        emit LoanApproval(user, loanAmount);
    }

    function repayLoan(uint256 amount) public {
        require(users[msg.sender].isLoanActive, "No active loan found");
        require(amount <= users[msg.sender].loanAmount, "Repayment amount exceeds loan amount");
        users[msg.sender].loanAmount -= amount;
        users[msg.sender].interestAccrued += calculateInterest(amount);
        totalInterestAccrued += calculateInterest(amount);
        emit LoanRepayment(msg.sender, amount);
    }

    function calculateInterest(uint256 amount) internal view returns (uint256) {
        return amount.mul(INTEREST_RATE).div(100);
    }

    function getBalance() public view returns (uint256) {
        return users[msg.sender].balance;
    }

    function getLoanAmount() public view returns(uint256) {
        return users[msg.sender].loanAmount;
    }

    function getInterestAccrued() public view returns (uint256) {
        return users[msg.sender].interestAccrued;
    }

    function getTotalDeposits() public view returns (uint256) {
        return totalDeposits;
    }

    function getTotalLoans() public view returns (uint256) {
        return totalLoans;
    }

    function getTotalInterestAccrued() public view returns (uint256) {
        return totalInterestAccrued;
    }

    function closeAccount() public onlyUser(msg.sender) {
        require(users[msg.sender].balance.eq(0), "Balance is not zero");
        require(users[msg.sender].loanAmount.eq(0), "Loan amount is not zero");
        require(users[msg.sender].interestAccrued.eq(0), "Interest accrued is not zero");
        delete users[msg.sender];
    }

    function destroy() public onlyOwner {
        selfdestruct(owner);
    }
}
