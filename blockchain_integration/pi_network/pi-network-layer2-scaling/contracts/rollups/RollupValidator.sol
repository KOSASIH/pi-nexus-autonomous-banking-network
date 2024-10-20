// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./Rollup.sol";

contract RollupValidator {
    Rollup public rollup;

    event TransactionValidated(bytes32 indexed stateRoot, address indexed sender);

    constructor(address _rollup) {
        rollup = Rollup(_rollup);
    }

    function validateTransaction(bytes32 _stateRoot, address _sender) external {
        require(rollup.getCurrentStateRoot() == _stateRoot, "Invalid state root");
        emit TransactionValidated(_stateRoot, _sender);
    }
}
