// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract InsuranceContract {
    struct Policy {
        address insured;
        uint256 premium;
        uint256 coverageAmount;
        bool isActive;
    }

    mapping(address => Policy) public policies;

    event PolicyCreated(address indexed insured, uint256 premium, uint256 coverageAmount);
    event PolicyClaimed(address indexed insured, uint256 amount);

    function createPolicy(uint256 premium, uint256 coverageAmount) external payable {
        require(msg.value == premium, "Premium must be paid");
        policies[msg.sender] = Policy(msg.sender, premium, coverageAmount, true);
        emit PolicyCreated(msg.sender, premium, coverageAmount);
    }

    function claim() external {
        Policy storage policy = policies[msg.sender];
        require(policy.isActive, "No active policy");
        require(address(this).balance >= policy.coverageAmount, "Insufficient funds for claim");

        policy.isActive = false;
        payable(msg.sender).transfer(policy.coverageAmount);
        emit PolicyClaimed(msg.sender, policy.coverageAmount);
    }
}
