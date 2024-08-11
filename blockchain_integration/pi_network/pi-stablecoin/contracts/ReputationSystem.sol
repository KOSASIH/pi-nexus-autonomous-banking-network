pragma solidity ^0.8.0;

import "./ReputationValidator.sol";
import "./ReputationScoreCalculator.sol";

contract ReputationSystem {
    // Mapping of user addresses to their reputation scores
    mapping (address => uint256) public reputationScores;

    // Mapping of user addresses to their reputation validators
    mapping (address => ReputationValidator[]) public reputationValidators;

    // Event emitted when a user's reputation score is updated
    event ReputationScoreUpdated(address indexed user, uint256 newScore);

    // Event emitted when a user's reputation validator is added
    event ReputationValidatorAdded(address indexed user, ReputationValidator validator);

    // Event emitted when a user's reputation validator is removed
    event ReputationValidatorRemoved(address indexed user, ReputationValidator validator);

    // Constructor
    constructor() public {
        // Initialize the reputation score calculator
        reputationScoreCalculator = new ReputationScoreCalculator();
    }

    // Function to add a reputation validator for a user
    function addReputationValidator(address user, ReputationValidator validator) public {
        // Check if the user is valid
        require(user != address(0), "Invalid user");

        // Add the reputation validator to the user's list
        reputationValidators[user].push(validator);

        // Emit the ReputationValidatorAdded event
        emit ReputationValidatorAdded(user, validator);
    }

    // Function to remove a reputation validator for a user
    function removeReputationValidator(address user, ReputationValidator validator) public {
        // Check if the user is valid
        require(user != address(0), "Invalid user");

        // Remove the reputation validator from the user's list
        for (uint256 i = 0; i < reputationValidators[user].length; i++) {
            if (reputationValidators[user][i] == validator) {
                reputationValidators[user][i] = reputationValidators[user][reputationValidators[user].length - 1];
                reputationValidators[user].pop();
                break;
            }
        }

        // Emit the ReputationValidatorRemoved event
        emit ReputationValidatorRemoved(user, validator);
    }

    // Function to update a user's reputation score
    function updateReputationScore(address user) public {
        // Check if the user is valid
        require(user != address(0), "Invalid user");

        // Calculate the new reputation score
        uint256 newScore = reputationScoreCalculator.calculateReputationScore(user, reputationValidators[user]);

        // Update the user's reputation score
        reputationScores[user] = newScore;

        // Emit the ReputationScoreUpdated event
        emit ReputationScoreUpdated(user, newScore);
    }
}
