pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";

contract PiUSDTreasury {
    using SafeERC20 for address;
    using SafeMath for uint256;

    // Events
    event NewTreasuryManager(address indexed newManager);
    event TreasuryFunded(uint256 amount);
    event TreasuryWithdrawal(uint256 amount);
    event ReserveRatioUpdated(uint256 newRatio);
    event InterestRateUpdated(uint256 newRate);

    // Constants
    uint256 public constant RESERVE_RATIO_DENOMINATOR = 10000; // 100% = 10000
    uint256 public constant INTEREST_RATE_DENOMINATOR = 10000; // 100% = 10000

    // State variables
    address public treasuryManager; // The address of the treasury manager
    address public piUSDToken; // The address of the PiUSD token contract
    uint256 public reserveRatio; // The reserve ratio as a percentage (e.g., 20% = 2000)
    uint256 public interestRate; // The interest rate as a percentage (e.g., 5% = 500)
    uint256 public treasuryBalance; // The current balance of the treasury
    mapping(address => uint256) public userBalances; // Mapping of user addresses to their balances

    // Modifiers
    modifier onlyTreasuryManager() {
        require(msg.sender == treasuryManager, "Only the treasury manager can call this function");
        _;
    }

    // Constructor
    constructor(address _piUSDToken) public {
        piUSDToken = _piUSDToken;
        treasuryManager = msg.sender;
        reserveRatio = 2000; // Initial reserve ratio: 20%
        interestRate = 500; // Initial interest rate: 5%
    }

    // Functions
    function fundTreasury(uint256 _amount) public onlyTreasuryManager {
        require(_amount > 0, "Amount must be greater than 0");
        treasuryBalance = treasuryBalance.add(_amount);
        emit TreasuryFunded(_amount);
    }

    function withdrawFromTreasury(uint256 _amount) public onlyTreasuryManager {
        require(_amount > 0, "Amount must be greater than 0");
        require(treasuryBalance >= _amount, "Insufficient treasury balance");
        treasuryBalance = treasuryBalance.sub(_amount);
        emit TreasuryWithdrawal(_amount);
    }

    function updateReserveRatio(uint256 _newRatio) public onlyTreasuryManager {
        require(_newRatio > 0, "New reserve ratio must be greater than 0");
        reserveRatio = _newRatio;
        emit ReserveRatioUpdated(_newRatio);
    }

    function updateInterestRate(uint256 _newRate) public onlyTreasuryManager {
        require(_newRate > 0, "New interest rate must be greater than 0");
        interestRate = _newRate;
        emit InterestRateUpdated(_newRate);
    }

    function deposit(uint256 _amount) public {
        require(_amount > 0, "Amount must be greater than 0");
        userBalances[msg.sender] = userBalances[msg.sender].add(_amount);
        treasuryBalance = treasuryBalance.add(_amount);
        emit TreasuryFunded(_amount);
    }

    function withdraw(uint256 _amount) public {
        require(_amount > 0, "Amount must be greater than 0");
        require(userBalances[msg.sender] >= _amount, "Insufficient user balance");
        userBalances[msg.sender] = userBalances[msg.sender].sub(_amount);
        treasuryBalance = treasuryBalance.sub(_amount);
        emit TreasuryWithdrawal(_amount);
    }

    function getTreasuryBalance() public view returns (uint256) {
        return treasuryBalance;
    }

    function getUserBalance(address _user) public view returns (uint256) {
        return userBalances[_user];
    }

    function getReserveRatio() public view returns (uint256) {
        return reserveRatio;
    }

    function getInterestRate() public view returns (uint256) {
        return interestRate;
    }
}
