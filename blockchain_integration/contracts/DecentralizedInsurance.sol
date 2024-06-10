pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract DecentralizedInsurance {
    // Mapping of policyholders to policy details
    mapping (address => Policy) public policies;

    // Event emitted when a new policy is created
    event PolicyCreated(address policyholder, uint256 policyId, uint256 premium, uint256 coverage);

    // Function to create a new policy
    function createPolicy(uint256 _premium, uint256 _coverage) public {
        // Check if premium and coverage are valid
        require(_premium > 0, "Invalid premium");
        require(_coverage > 0, "Invalid coverage");

        // Create new policy
        uint256 policyId = policies.length();
        policies[policyId] = Policy(msg.sender, policyId, _premium, _coverage);

        // Emit policy created event
        emit PolicyCreated(msg.sender, policyId, _premium, _coverage);
    }

    // Function to file a claim
    function fileClaim(uint256 _policyId, uint256 _claimAmount) public {
        // Check if policy exists
        require(policies[_policyId].policyholder == msg.sender, "Policy does not exist");

        // Check if claim amount is valid
        require(_claimAmount <= policies[_policyId].coverage, "Invalid claim amount");

        // Pay out claim amount to policyholder
        msg.sender.transfer(_claimAmount);
    }
}

struct Policy {
    address policyholder;
    uint256 policyId;
    uint256 premium;
    uint256 coverage;
}
