pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract Staking {
    using SafeMath for uint256;

    // Mapping of stakers
    mapping (address => uint256) public stakers;

    // Event emitted when a staker is added
    event StakerAdded(address indexed staker);

    // Event emitted when a staker is removed
    event StakerRemoved(address indexed staker);

    // Event emitted when a staker's balance is updated
    event StakerBalanceUpdated(address indexed staker, uint256 balance);

    // Function to add a staker
    function addStaker(address staker) public {
        // Only allow adding stakers by authorized addresses
        require(msg.sender == governanceContract, "Only governance contract can add stakers");

        // Add staker
        stakers[staker] = 0;
        emit StakerAdded(staker);
    }

    // Function to remove a staker
    function removeStaker(address staker) public {
        // Only allow removing stakers by authorized addresses
        require(msg.sender == governanceContract, "Only governance contract can remove stakers");

        // Remove staker
        delete stakers[staker];
        emit StakerRemoved(staker);
    }

    // Function to update a staker's balance
    function updateStakerBalance(address staker, uint256 balance) public {
        // Only allow updating staker balances by authorized addresses
        require(msg.sender == governanceContract, "Only governance contract can update staker balances");

        // Update staker balance
        stakers[staker] = balance;
        emit StakerBalanceUpdated(staker, balance);
    }
}
