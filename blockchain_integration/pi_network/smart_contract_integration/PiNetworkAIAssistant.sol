pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/AIAssistant/AIAssistant.sol";

contract PiNetworkAIAssistant is AIAssistant {
    // Mapping of user addresses to their AI assistants
    mapping (address => AIAssistantData) public aiAssistants;

    // Struct to represent an AI assistant
    struct AIAssistantData {
        string assistantType;
        string assistantData;
    }

    // Event emitted when a new AI assistant is created
    event AIAssistantCreatedEvent(address indexed user, AIAssistantData assistant);

    // Function to create a new AI assistant
    function createAIAssistant(string memory assistantType, string memory assistantData) public {
        AIAssistantData storage assistant = aiAssistants[msg.sender];
        assistant.assistantType = assistantType;
        assistant.assistantData = assistantData;
        emit AIAssistantCreatedEvent(msg.sender, assistant);
    }

    // Function to get an AI assistant
    function getAIAssistant(address user) public view returns (AIAssistantData memory) {
        return aiAssistants[user];
    }
}
