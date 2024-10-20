// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract RollupManager {
    mapping(address => bool) public operators;
    address public owner;

    event OperatorAdded(address indexed operator);
    event OperatorRemoved(address indexed operator);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function addOperator(address _operator) external onlyOwner {
        require(!operators[_operator], "Already an operator");
        operators[_operator] = true;
        emit OperatorAdded(_operator);
    }

    function removeOperator(address _operator) external onlyOwner {
        require(operators[_operator], "Not an operator");
        operators[_operator] = false;
        emit OperatorRemoved(_operator);
    }

    function isOperator(address _address) external view returns (bool) {
        return operators[_address];
    }
}
