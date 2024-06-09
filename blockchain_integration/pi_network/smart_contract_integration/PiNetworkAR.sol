pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/AR/AR.sol";

contract PiNetworkAR is AR {
    // Mapping of user addresses to their AR experiences
    mapping (address => ARExperience) public arExperiences;

    // Struct to represent an AR experience
    struct ARExperience {
        string experienceType;
        string experienceData;
    }

    // Event emitted when a new AR experience is created
    event ARExperienceCreatedEvent(address indexed user, ARExperience experience);

    // Function to create a new AR experience
    function createARExperience(string memory experienceType, string memory experienceData) public {
        ARExperience storage experience = arExperiences[msg.sender];
        experience.experienceType = experienceType;
        experience.experienceData = experienceData;
        emit ARExperienceCreatedEvent(msg.sender, experience);
    }

    // Function to get an AR experience
    function getARExperience(address user) public view returns (ARExperience memory) {
        return arExperiences[user];
    }
}
