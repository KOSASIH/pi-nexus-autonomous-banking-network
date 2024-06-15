pragma solidity ^0.8.0;

import "./PIBank.sol";

contract PIBankInsurance {
    // Mapping of insurance policies
    mapping(address => InsurancePolicy) public insurancePolicies;

    // Event
    event NewInsurancePolicy(address indexed user, uint256 amount);

    // Function
    function purchaseInsurance(address user, uint256 amount) public {
        // Create a new insurance policy
        InsurancePolicy policy = InsurancePolicy(user, amount);
        insurancePolicies[user] = policy;
        emit NewInsurancePolicy(user, amount);
    }
}
