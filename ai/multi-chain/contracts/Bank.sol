pragma solidity ^0.8.0;

import "./RelayHub.sol";

contract Bank {
    // Address of the RelayHub contract
    address public relayHubAddress;

    // Event emitted when a new account is created
    event AccountCreated(address indexed account);

    // Constructor
    constructor(address relayHubAddress) {
        this.relayHubAddress = relayHubAddress;
    }

    // Create a new account
    function createAccount() public {
        // Interact with the RelayHub contract to create a new account
        RelayHub(relayHubAddress).createAccount();
    }

    // Deposit funds
    function deposit(uint256 amount) public {
        // Interact with the RelayHub contract to deposit funds
        RelayHub(relayHubAddress).deposit(msg.sender, address(this), amount);
    }

    // Withdraw funds
    function withdraw(uint256 amount) public {
        // Interact with the RelayHub contract to withdraw funds
        RelayHub(relayHubAddress).withdraw(msg.sender, address(this), amount);
    }
}
