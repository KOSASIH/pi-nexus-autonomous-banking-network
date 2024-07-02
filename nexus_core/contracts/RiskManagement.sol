pragma solidity ^0.8.0;

contract RiskManagement {
    mapping (address => uint256) public riskScores;

    constructor() {
        // Initialize risk score mapping
    }

    function updateRiskScore(address account, uint256 score) public {
        riskScores[account] = score;
    }

    function getRiskScore(address account) public view returns (uint256) {
        return riskScores[account];
    }
}
