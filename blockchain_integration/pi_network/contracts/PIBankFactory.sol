pragma solidity ^0.8.0;

import "./PIBank.sol";

contract PIBankFactory {
    // Mapping of PIBank instances
    mapping (address => PIBank) public banks;

    // Create a new PIBank instance
    function createPIBank() public {
        PIBank newBank = new PIBank();
        banks[msg.sender] = newBank;
    }

    // Get a PIBank instance
    function getPIBank(address owner) public view returns (PIBank) {
        return banks[owner];
    }
}
