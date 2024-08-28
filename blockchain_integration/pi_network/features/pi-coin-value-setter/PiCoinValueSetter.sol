pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract PiCoinValueSetter {
    // Set the target value of Pi Coin
    uint256 public targetValue = 314159000000000000000; // $314.159

    // Set the mainnet launch timestamp
    uint256 public mainnetLaunchTimestamp;

    // Event emitted when the mainnet is launched
    event MainnetLaunched();

    // Event emitted when the Pi Coin value is updated
    event PiCoinValueUpdated(uint256 newValue);

    // Mapping of Pi Coin values
    mapping (address => uint256) public piCoinValues;

    // Function to set the value of Pi Coin when the mainnet is launched
    function setPiCoinValue() public {
        // Check if the mainnet has been launched
        require(mainnetLaunchTimestamp != 0, "Mainnet has not been launched");

        // Set the value of Pi Coin to the target value
        for (address user in piCoinValues) {
            piCoinValues[user] = targetValue;
            emit PiCoinValueUpdated(targetValue);
        }

        // Emit the MainnetLaunched event
        emit MainnetLaunched();
    }

    // Function to set the mainnet launch timestamp
    function setMainnetLaunchTimestamp(uint256 _timestamp) public {
        mainnetLaunchTimestamp = _timestamp;
    }

    // Function to get the value of a Pi Coin
    function getPiCoinValue(address _user) public view returns (uint256) {
        return piCoinValues[_user];
    }

    // Function to update the value of a Pi Coin
    function updatePiCoinValue(address _user, uint256 _newValue) public {
        require(piCoinValues[_user] > 0, "Pi Coin value must be set before updating");
        piCoinValues[_user] = _newValue;
        emit PiCoinValueUpdated(_newValue);
    }

    // Function to calculate the total value of all Pi Coins
    function getTotalPiCoinValue() public view returns (uint256) {
        uint256 totalValue = 0;
        for (address user in piCoinValues) {
            totalValue = SafeMath.add(totalValue, piCoinValues[user]);
        }
        return totalValue;
    }
}
