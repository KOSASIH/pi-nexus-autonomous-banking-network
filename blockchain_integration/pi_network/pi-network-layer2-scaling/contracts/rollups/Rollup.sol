// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./RollupManager.sol";

contract Rollup {
    RollupManager public manager;
    bytes32 public currentStateRoot;
    uint256 public batchCount;

    event BatchCreated(bytes32 indexed stateRoot, uint256 indexed batchId);

    constructor(address _manager) {
        manager = RollupManager(_manager);
    }

    function createBatch(bytes32 _stateRoot) external {
        require(manager.isOperator(msg.sender), "Not an operator");
        currentStateRoot = _stateRoot;
        batchCount++;
        emit BatchCreated(_stateRoot, batchCount);
    }

    function getCurrentStateRoot() external view returns (bytes32) {
        return currentStateRoot;
    }
}
