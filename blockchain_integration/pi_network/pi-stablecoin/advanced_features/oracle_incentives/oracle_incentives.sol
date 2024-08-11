pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract OracleIncentives {
    // Mapping of oracle nodes
    mapping (address => OracleNode) public oracleNodes;

    // Mapping of oracle rewards
    mapping (address => uint256) public oracleRewards;

    // Event emitted when an oracle node submits a price update
    event PriceUpdate(address indexed oracle, uint256 price);

    // Event emitted when an oracle node is rewarded
    event Reward(address indexed oracle, uint256 amount);

    // Struct to represent an oracle node
    struct OracleNode {
        address node;
        uint256 reputation;
        uint256 lastUpdate;
    }

    // Function to register an oracle node
    function registerOracleNode(address node) public {
        // Create a new oracle node
        OracleNode storage oracleNode = oracleNodes[node];
        oracleNode.node = node;
        oracleNode.reputation = 0;
        oracleNode.lastUpdate = block.timestamp;
    }

    // Function to submit a price update
    function submitPriceUpdate(address oracle, uint256 price) public {
        // Check if the oracle node is registered
        require(oracleNodes[oracle].node != address(0), "Oracle node not registered");

        // Update the oracle node's last update timestamp
        oracleNodes[oracle].lastUpdate = block.timestamp;

        // Emit the PriceUpdate event
        emit PriceUpdate(oracle, price);

        // Calculate the reward for the oracle node
        uint256 reward = calculateReward(oracle);

        // Update the oracle node's reputation
        oracleNodes[oracle].reputation += reward;

        // Emit the Reward event
        emit Reward(oracle, reward);
    }

    // Function to calculate the reward for an oracle node
    function calculateReward(address oracle) internal returns (uint256) {
        // Calculate the reward based on the oracle node's reputation and last update timestamp
        uint256 reward = oracleNodes[oracle].reputation * (block.timestamp - oracleNodes[oracle].lastUpdate);
        return reward;
    }

    // Function to get the reward for an oracle node
    function getReward(address oracle) public view returns (uint256) {
        return oracleRewards[oracle];
    }

    // Function to claim rewards
    function claimRewards(address oracle) public {
        // Check if the oracle node has rewards to claim
        require(oracleRewards[oracle] > 0, "No rewards to claim");

        // Transfer the rewards to the oracle node
        SafeERC20.transfer(piStablecoin, oracle, oracleRewards[oracle]);

        // Reset the oracle node's rewards
        oracleRewards[oracle] = 0;
    }
}
