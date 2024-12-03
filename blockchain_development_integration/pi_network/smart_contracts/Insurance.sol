// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Insurance {
    struct Policy {
        address policyHolder;
        uint256 premium;
        uint256 coverageAmount;
        bool isActive;
    }

    mapping(uint256 => Policy) public policies;
    uint256 public policyCount;

    function purchasePolicy(uint256 premium, uint256 coverageAmount) external payable {
        require(msg.value == premium, "Incorrect premium amount");
        policyCount++;
        policies[policyCount] = Policy(msg.sender, premium, coverageAmount, true);
    }

    function claimPolicy(uint256 policyId) external {
        Policy storage policy = policies[policyId];
        require(policy.policyHolder == msg.sender, "Not the policy holder");
        require(policy.isActive, "Policy is not active");

        policy.isActive = false;
        payable(msg.sender).transfer(policy.coverageAmount);
    }
}
