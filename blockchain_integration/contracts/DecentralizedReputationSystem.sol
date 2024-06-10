pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract DecentralizedReputationSystem {
    // Mapping of user addresses to reputation scores
    mapping (address => uint256) public reputationScores;

    // Event emitted when a user's reputation score is updated
    event ReputationScoreUpdated(address user, uint256 newScore);

    // Function to update a user's reputation score
    function updateReputationScore(address _user, uint256 _newScore) public {
        // Check if user exists
        require(reputationScores[_user] != 0, "User does not exist");

        // Update reputation score
        reputationScores[_user] = _newScore;

        // Emit reputation score updated event
        emit ReputationScoreUpdated(_user, _newScore);
    }

    // Function to get a user's reputation score
    function getReputationScore(address _user) public view returns (uint256) {
        return reputationScores[_user];
    }
}
