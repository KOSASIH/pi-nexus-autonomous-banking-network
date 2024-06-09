pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/VR/VR.sol";

contract PiNetworkVR is VR {
    // Mapping of user addresses to their VR experiences
    mapping (address => VRExperience) public vrExperiences;

    // Struct to represent a VR experience
    struct VRExperience {
        string experienceType;
        string experienceData;
    }

    // Event emitted when a new VR experience is created
    event VRExperienceCreatedEvent(address indexed user, VRExperience experience);

    // Function to create a new VR experience
    function createVRExperience(string memory experienceType, string memory experienceData) public {
        VRExperience storage experience = vrExperiences[msg.sender];
        experience.experienceType = experienceType;
        experience.experienceData = experienceData;
        emit VRExperienceCreatedEvent(msg.sender, experience);
    }

    // Function to get a VR experience
    function getVRExperience(address user) public view returns (VRExperience memory) {
        return vrExperiences[user];
    }
}
