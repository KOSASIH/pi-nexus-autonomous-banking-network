pragma solidity ^0.8.0;

contract NexusToken {
    address private owner;
    uint public totalSupply;

    constructor() public {
        owner = msg.sender;
        totalSupply = 1000000;
    }

    function transfer(address recipient, uint amount) public {
        require(msg.sender == owner, "Only the owner can transfer tokens");
        require(amount <= totalSupply, "Insufficient tokens");
        totalSupply -= amount;
        recipient.transfer(amount);
    }
}
