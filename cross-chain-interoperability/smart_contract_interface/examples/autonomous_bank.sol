// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "./smart_contract_interface.sol";

contract AutonomousBank is IBank {
    // Mapping of account addresses to balances
    mapping(address => uint256) public balances;

    // Function to get the current balance of an account
    function getBalance(address account) external view override returns (uint256) {
        return balances[account];
    }

    // Function to deposit funds into an account
    function deposit(address account, uint256 amount) external override {
        balances[account] += amount;
    }

    // Function to withdraw funds from an account
    function withdraw(address account, uint256 amount) external override {
        require(balances[account] >= amount, "Insufficient funds");
        balances[account] -= amount;
    }

    // Function to transfer funds from one account to another
    function transfer(address from, address to, uint256 amount) external override {
        require(balances[from] >= amount, "Insufficient funds");
        balances[from] -= amount;
        balances[to] += amount;
    }
}
