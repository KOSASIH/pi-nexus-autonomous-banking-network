pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

contract Interest is Ownable, Pausable {
    // Interest rate variables
    uint256 public constant BASE_RATE = 50; // Base interest rate in basis points (0.50%)
    uint256 public constant MULTIPLIER = 2; // Interest rate multiplier
    uint256 public constant DIVISOR = 10000; // Divisor for converting basis points to decimal form

    // User interest variables
    mapping(address => uint256) public userInterest;

    // Event for when interest is calculated
    event InterestCalculated(address indexed user, uint256 interest);

    // Constructor
    constructor() ERC20("Interest", "INT") {}

    // Function to calculate interest for a user
    function calculateInterest(address user) public onlyOwner {
        require(!paused(), "Contract is paused");

        // Calculate the interest rate for the user
        uint256 interestRate = BASE_RATE + (userInterest[user] * MULTIPLIER) / DIVISOR;

        // Calculate the interest amount
        uint256 interest = balanceOf(user) * interestRate / DIVISOR;

        // Update the user's interest balance
        userInterest[user] += interest;

        // Emit an event for when interest is calculated
        emit InterestCalculated(user, interest);
    }

    // Function to update the user's interest balance
    function updateUserInterest(address user, uint256 newInterest) public onlyOwner {
        require(!paused(), "Contract is paused");
        userInterest[user] = newInterest;
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
