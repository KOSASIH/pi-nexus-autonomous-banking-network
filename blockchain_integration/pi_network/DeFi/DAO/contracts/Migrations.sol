pragma solidity ^0.8.0;

contract Migrations {
    address public owner;
    uint public lastCompletedMigration;

    constructor() public {
        owner = msg.sender;
        lastCompletedMigration = 0;
    }

    function setCompleted(uint completed) public {
        require(msg.sender == owner, "Only the owner can set the completed migration");
        lastCompletedMigration = completed;
    }

    function upgrade(address newAddress) public {
        require(msg.sender == owner, "Only the owner can upgrade the contract");
        Migrations upgraded = Migrations(newAddress);
        upgraded.setCompleted(lastCompletedMigration);
    }
}
