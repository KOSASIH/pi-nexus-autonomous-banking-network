pragma solidity ^0.8.0;

import "./ReputationSystem.sol";

contract InsuranceFund {
    // Mapping of user addresses to their insurance fund contributions
    mapping (address => uint256) public insuranceFundContributions;

    // Event emitted when a user's insurance fund contribution changes
    event InsuranceFundContributionChanged(address user, uint256 newContribution);

    // Constructor
    constructor() public {
        // Initialize the insurance fund contributions for all users to 0
        for (address user in ReputationSystem.allUsers) {
            insuranceFundContributions[user] = 0;
        }
    }

    // Function to contribute to the insurance fund
    function contributeToInsuranceFund(address user, uint256 amount) public {
        // Update the user's insurance fund contribution
        insuranceFundContributions[user] += amount;
        emit InsuranceFundContributionChanged(user, insuranceFundContributions[user]);
    }

    // Function to get a user's insurance fund contribution
    function getInsuranceFundContribution(address user) public view returns (uint256) {
        return insuranceFundContributions[user];
    }
}
