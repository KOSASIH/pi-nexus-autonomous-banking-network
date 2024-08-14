pragma solidity ^0.6.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract PiUSD {
    using SafeMath for uint256;

    // Mapping of user addresses to PiUSD token balances
    mapping (address => uint256) public balances;

    // Mapping of user addresses to collateral balances
    mapping (address => uint256) public collateral;

    // Total supply of PiUSD tokens
    uint256 public totalSupply;

    // Interest rate (in basis points)
    uint256 public interestRate;

    // Event emitted when PiUSD tokens are minted
    event Mint(address indexed user, uint256 amount);

    // Event emitted when PiUSD tokens are burned
    event Burn(address indexed user, uint256 amount);

    // Event emitted when collateral is deposited
    event Deposit(address indexed user, uint256 amount);

    // Event emitted when collateral is withdrawn
    event Withdrawal(address indexed user, uint256 amount);

    // Event emitted when the interest rate is updated
    event InterestRateUpdated(uint256 newInterestRate);

    // Function to mint PiUSD tokens
    function mint(uint256 amount) public {
        // Check if the user has sufficient collateral
        require(collateral[msg.sender] >= amount, "Insufficient collateral");

        // Mint the PiUSD tokens
        balances[msg.sender] = balances[msg.sender].add(amount);
        totalSupply = totalSupply.add(amount);

        // Emit the Mint event
        emit Mint(msg.sender, amount);
    }

    // Function to burn PiUSD tokens
    function burn(uint256 amount) public {
        // Check if the user has sufficient PiUSD tokens
        require(balances[msg.sender] >= amount, "Insufficient PiUSD tokens");

        // Burn the PiUSD tokens
        balances[msg.sender] = balances[msg.sender].sub(amount);
        totalSupply = totalSupply.sub(amount);

        // Emit the Burn event
        emit Burn(msg.sender, amount);
    }

    // Function to deposit collateral
    function deposit(uint256 amount) public {
        // Update the user's collateral balance
        collateral[msg.sender] = collateral[msg.sender].add(amount);

        // Emit the Deposit event
        emit Deposit(msg.sender, amount);
    }

    // Function to withdraw collateral
    function withdraw(uint256 amount) public {
        // Check if the user has sufficient collateral
        require(collateral[msg.sender] >= amount, "Insufficient collateral");

        // Update the user's collateral balance
        collateral[msg.sender] = collateral[msg.sender].sub(amount);

        // Emit the Withdrawal event
        emit Withdrawal(msg.sender, amount);
    }

    // Function to update the interest rate
    function updateInterestRate(uint256 newInterestRate) public {
        // Check if the user is authorized to update the interest rate
        require(msg.sender == governanceAddress, "Unauthorized access");

        // Update the interest rate
        interestRate = newInterestRate;

        // Emit the InterestRateUpdated event
        emit InterestRateUpdated(newInterestRate);
    }
}
