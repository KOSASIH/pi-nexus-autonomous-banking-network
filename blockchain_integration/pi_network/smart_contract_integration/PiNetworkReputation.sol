pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract PiNetworkReputation {
    // Mapping of user addresses to their reputation scores
    mapping (address => uint256) public reputationScores;

    // Event emitted when a user's reputation score is updated
    event ReputationUpdateEvent(address indexed user, uint256 score);

    // Function to update a user's reputation score
    function updateReputationScore(address user, uint256 score) public {
        reputationScores[user] = SafeMath.add(reputationScores[user], score);
        emit ReputationUpdateEvent(user, reputationScores[user]);
    }

    // Function to get a user's reputation score
    function getReputationScore(address user) public view returns (uint256) {
        return reputationScores[user];
    }
}
