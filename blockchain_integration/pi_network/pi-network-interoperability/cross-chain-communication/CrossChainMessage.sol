pragma solidity ^0.8.0;

contract CrossChainMessage {
    // Mapping of message IDs to their respective messages
    mapping (bytes32 => bytes) public messages;

    // Event emitted when a message is sent
    event MessageSent(bytes32 indexed messageId, bytes message);

    // Function to send a message
    function sendMessage(bytes32 messageId, bytes message) public {
        // Store the message in the messages mapping
        messages[messageId] = message;

        // Emit the MessageSent event
        emit MessageSent(messageId, message);
    }
}
