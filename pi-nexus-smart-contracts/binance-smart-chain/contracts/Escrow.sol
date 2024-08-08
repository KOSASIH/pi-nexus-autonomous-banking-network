// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

contract Escrow is ReentrancyGuard {
    struct EscrowAccount {
        address payable owner;
        address payable beneficiary;
        uint256 amount;
        bool released;
    }

    mapping(address => EscrowAccount) public escrowAccounts;

    function createEscrowAccount(address beneficiary, uint256 amount) public {
        require(beneficiary != address(0), "Invalid beneficiary address");
        require(amount > 0, "Invalid amount");

        EscrowAccount storage account = escrowAccounts[msg.sender];
        account.owner = payable(msg.sender);
        account.beneficiary = beneficiary;
        account.amount = amount;
        account.released = false;
    }

    function releaseFunds(uint256 index) public {
        require(index > 0, "Invalid index");

        EscrowAccount storage account = escrowAccounts[msg.sender];
        require(account.owner == msg.sender, "Not the owner of the account");
        require(!account.released, "Funds already released");

        account.beneficiary.transfer(account.amount);
        account.released = true;
    }

    function dispute(uint256 index) public {
        require(index > 0, "Invalid index");

        EscrowAccount storage account = escrowAccounts[msg.sender];
        require(account.owner == msg.sender, "Not the owner of the account");

        // Implement dispute resolution logic here
    }
}
