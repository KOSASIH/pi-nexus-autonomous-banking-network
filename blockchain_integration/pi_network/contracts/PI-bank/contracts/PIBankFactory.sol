pragma solidity ^0.8.0;

import "./PIBank.sol";

contract PIBankFactory {
    // Mapping of PIBank instances
    mapping(address => PIBank) public pibanks;

    // Event
    event NewPIBank(address indexed pibankAddress);

    // Function
    function createPIBank() public {
        // Create a new PIBank instance
        PIBank pibank = new PIBank();
        pibanks[msg.sender] = pibank;
        emit NewPIBank(address(pibank));
    }
}
