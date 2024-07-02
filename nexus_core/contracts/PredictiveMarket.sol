pragma solidity ^0.8.0;

contract PredictiveMarket {
    mapping (address => mapping (string => uint256)) public predictions;

    constructor() {
        // Initialize prediction mapping
    }

    function makePrediction(string memory event, uint256 outcome) public {
        predictions[msg.sender][event] = outcome;
    }

    function resolveEvent(string memory event, uint256 outcome) public {
        // Resolve event logic
    }

    function getPredictions(address account) public view returns (mapping (string => uint256)) {
        return predictions[account];
    }
}
