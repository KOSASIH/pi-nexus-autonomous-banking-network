pragma solidity ^0.8.0;

contract BlockchainCommunication {
    address private owner;
    mapping (address => string) public messages;

    constructor() public {
        owner = msg.sender;
    }

    function sendMessage(string memory _message) public {
        require(msg.sender == owner, "Only the owner can send messages");
        messages[msg.sender] = _message;
    }

    function receiveMessage() public view returns (string memory) {
        return messages[msg.sender];
    }
}

// Example usage:
contract MyContract {
    BlockchainCommunication public communication;

    constructor() public {
        communication = new BlockchainCommunication();
    }

    function sendMessage(string memory _message) public {
        communication.sendMessage(_message);
    }

    function receiveMessage() public view returns (string memory) {
        return communication.receiveMessage();
    }
}
