pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract PiStableCoin {
    // Mapping of user balances
    mapping (address => uint256) public balances;

    // Total supply of Pi Coin
    uint256 public totalSupply;

    // Reserve of assets
    uint256 public reserve;

    // Oracle service
    address public oracle;

    // Governance mechanism
    address public governance;

    // Price feed contract
    address public priceFeed;

    // Rebalancing threshold (e.g. 1% deviation from target price)
    uint256 public rebalancingThreshold;

    // Target price of Pi Coin (e.g. $314.159)
    uint256 public targetPrice;

    // Event emitted when the value of Pi Coin changes
    event ValueChanged(uint256 newValue);

    // Event emitted when the reserve is updated
    event ReserveUpdated(uint256 newReserve);

    // Event emitted when the governance mechanism updates the target price
    event TargetPriceUpdated(uint256 newTargetPrice);

    // Event emitted when the rebalancing threshold is updated
    event RebalancingThresholdUpdated(uint256 newThreshold);

    // Modifier to check if the caller is the governance mechanism
    modifier onlyGovernance {
        require(msg.sender == governance, "Only governance can call this function");
        _;
    }

    // Modifier to check if the caller is the oracle service
    modifier onlyOracle {
        require(msg.sender == oracle, "Only oracle can call this function");
        _;
    }

    // Function to mint new Pi Coin
    function mint(address _to, uint256 _amount) public {
        // Check if the reserve is sufficient
        require(reserve >= _amount, "Insufficient reserve");

        // Mint new Pi Coin
        balances[_to] += _amount;
        totalSupply += _amount;

        // Update the reserve
        reserve -= _amount;
    }

    // Function to burn Pi Coin
    function burn(address _from, uint256 _amount) public {
        // Check if the user has sufficient balance
        require(balances[_from] >= _amount, "Insufficient balance");

        // Burn Pi Coin
        balances[_from] -= _amount;
        totalSupply -= _amount;

        // Update the reserve
        reserve += _amount;
    }

    // Function to update the value of Pi Coin
    function updateValue(uint256 _newValue) public onlyOracle {
        // Check if the new value is within the rebalancing threshold
        require(SafeMath.abs(_newValue - targetPrice) <= rebalancingThreshold, "Value is outside rebalancing threshold");

        // Update the value of Pi Coin
        emit ValueChanged(_newValue);
    }

    // Function to update the target price
    function updateTargetPrice(uint256 _newTargetPrice) public onlyGovernance {
        // Update the target price
        targetPrice = _newTargetPrice;
        emit TargetPriceUpdated(_newTargetPrice);
    }

    // Function to update the rebalancing threshold
    function updateRebalancingThreshold(uint256 _newThreshold) public onlyGovernance {
        // Update the rebalancing threshold
        rebalancingThreshold = _newThreshold;
        emit RebalancingThresholdUpdated(_newThreshold);
    }

    // Function to update the reserve
    function updateReserve(uint256 _newReserve) public onlyGovernance {
        // Update the reserve
        reserve = _newReserve;
        emit ReserveUpdated(_newReserve);
    }

    // Function to get the current price of Pi Coin
    function getPrice() public view returns (uint256) {
        // Get the current price from the price feed contract
        return priceFeed.getPrice();
    }

    // Function to get the current reserve ratio
    function getReserveRatio() public view returns (uint256) {
        // Calculate the current reserve ratio
        return reserve * 100 / totalSupply;
    }
}
