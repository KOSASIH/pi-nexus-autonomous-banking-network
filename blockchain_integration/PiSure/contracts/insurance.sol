pragma solidity ^0.8.0;

import "./policy.sol";
import "./riskAssessment.sol";

contract Insurance {
    address private owner;
    mapping(address => Policy[]) public policies;
    RiskAssessment public riskAssessment;

    constructor() {
        owner = msg.sender;
        riskAssessment = new RiskAssessment();
    }

    function createPolicy(Policy _policy) public {
        require(msg.sender == owner, "Only the owner can create policies");
        policies[msg.sender].push(_policy);
    }

    function getPolicies() public view returns (Policy[] memory) {
        return policies[msg.sender];
    }

    function assessRisk(address _policyHolder, uint _amount) public {
        uint riskScore = riskAssessment.assessRisk(_policyHolder, _amount);
        if (riskScore > 50) {
            // High risk, reject policy
            revert("High risk, policy rejected");
        } else {
            // Low risk, approve policy
            createPolicy(Policy(_policyHolder, _amount));
        }
    }

    function payout(address _policyHolder, uint _amount) public {
        require(policies[_policyHolder].length > 0, "No policy found");
        // Payout logic goes here
    }
}
