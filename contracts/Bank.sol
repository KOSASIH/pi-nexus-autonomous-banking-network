// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

contract Bank is Ownable, Pausable {
    // Mapping to store user balances
    mapping(address => uint256) public balances;

    // Event to emit when a deposit is made
    event Deposit(address indexed user, uint256 amount);

    // Event to emit when a withdrawal is made
    event Withdrawal(address indexed user, uint256 amount);

    /**
     * @dev Constructor function to initialize the contract
     */
    constructor() {
        // Initialize the owner to the deployer address
        _setOwner(msg.sender);
    }

    /**
     * @dev Function to deposit funds into the contract
     * @param amount The amount of funds to deposit
     */
    function deposit(uint256 amount) public payable {
        // Require that the contract is not paused
        require(!paused(), "Contract is paused");

        // Add the amount to the user's balance
        balances[msg.sender] += amount;

        // Emit the Deposit event
        emit Deposit(msg.sender, amount);
    }

    /**
     * @dev Function to withdraw funds from the contract
     * @param amount The amount of funds to withdraw
     */
    function withdraw(uint256 amount) public {
        // Require that the contract is not paused
        require(!paused(), "Contract is paused");

        // Require that the user has sufficient balance
        require(balances[msg.sender] >= amount, "Insufficient balance");

        // Subtract the amount from the user's balance
        balances[msg.sender] -= amount;

        // Transfer the funds to the user
        payable(msg.sender).transfer(amount);

        // Emit the Withdrawal event
        emit Withdrawal(msg.sender, amount);
    }

    /**
     * @dev Function to check the balance of an account
     * @param user The address of the account to check
     * @return The balance of the account
     */
    function balanceOf(address user) public view returns (uint256) {
        return balances[user];
    }

    /**
     * @dev Function to pause the contract
     */
    function pause() public onlyOwner {
        _pause();
    }

    /**
     * @dev Function to unpause the contract
     */
    function unpause() public onlyOwner {
        _unpause();
    }
}
