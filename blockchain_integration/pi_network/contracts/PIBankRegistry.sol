pragma solidity ^0.8.0;

import "./PIBankFactory.sol";

contract PIBankRegistry {
    // Mapping of PIBankFactory instances
    mapping (address => PIBankFactory) public factories;

    // Create a new PIBankFactory instance
    function createPIBankFactory() public {
        PIBankFactory newFactory = new PIBankFactory();
        factories[msg.sender] = newFactory;
    }

    // Get a PIBankFactory instance
    function getPIBankFactory(address owner) public view returns (PIBankFactory) {
        return factories[owner];
    }
}
