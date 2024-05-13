pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract InsuranceContract {
    using IERC20 for IERC20Type;

    struct Policy {
        IERC20Type token;
        uint256 premium;
        uint256 coverageAmount;
        bool isActive;
        uint256 creationDate;
    }

    mapping(address => mapping(uint256 => Policy)) public policies;

    event PolicyCreated(address indexed policyOwner, uint256 indexed policyId, IERC20Type indexed token, uint256 premium, uint256 coverageAmount);
    event PolicyCancelled(address indexed policyOwner, uint256 indexed policyId);
    event PolicyClaim(address indexed policyOwner, uint256 indexed policyId, uint256 claimAmount);

    constructor() public {
        // Initialize contract
    }

    function createPolicy(IERC20Type _token, uint256 _premium, uint256 _coverageAmount) public {
        require(_token.transferFrom(msg.sender, address(this), _premium), "Not enough tokens to pay premium");

        Policy memory newPolicy;
        newPolicy.token = _token;
        newPolicy.premium = _premium;
        newPolicy.coverageAmount = _coverageAmount;
        newPolicy.isActive = true;
        newPolicy.creationDate = block.timestamp;

        uint256 policyId = policies[msg.sender].length;
        policies[msg.sender][policyId] = newPolicy;

        emit PolicyCreated(msg.sender, policyId, _token, _premium, _coverageAmount);
    }

    function cancelPolicy(uint256 _policyId) public {
        Policy storage policy = policies[msg.sender][_policyId];
        require(policy.isActive, "Policy is not active");

        policy.isActive = false;

        emit PolicyCancelled(msg.sender, _policyId);
    }

    function claimPolicy(uint256 _policyId, uint256 _claimAmount) public {
        Policy storage policy = policies[msg.sender][_policyId];
        require(policy.isActive, "Policy is not active");
        require(_claimAmount <= policy.coverageAmount, "Claim amount exceeds coverage amount");

        policy.token.transfer(msg.sender, _claimAmount);

        emit PolicyClaim(msg.sender, _policyId, _claimAmount);
    }

    function getPolicy(uint256 _policyId) public view returns (IERC20Type, uint256, uint256, bool, uint256) {
        Policy storage policy = policies[msg.sender][_policyId];
        return (policy.token, policy.premium, policy.coverageAmount, policy.isActive, policy.creationDate);
    }
}
