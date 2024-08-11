pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract ReputationSystem {
    // Mapping of node reputations
    mapping (address => uint256) public nodeReputations;

    // Function to update node reputation
    function updateNodeReputation(address node, uint256 reputation) public {
        // Update node reputation
        nodeReputations[node] = reputation;
    }
}
