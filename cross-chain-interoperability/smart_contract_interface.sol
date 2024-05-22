// smart_contract_interface.sol
// A Solidity file to define a common interface for smart contracts across different chains
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

interface IBank {
    // Function to get the current balance of an account
    function getBalance(address account) external view returns (uint256);

    // Function to deposit funds into an account
    function deposit(address account, uint256 amount) external;

    // Function to withdraw funds from an account
    function withdraw(address account, uint256 amount) external;

    // Function to transfer funds from one account to another
    function transfer(address from, address to, uint256 amount) external;
}
