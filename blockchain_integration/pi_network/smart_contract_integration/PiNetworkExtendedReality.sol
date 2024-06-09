pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/ExtendedReality/ExtendedReality.sol";

contract PiNetworkExtendedReality is ExtendedReality {
    // Mapping of user addresses to their extended reality experiences
    mapping (address => ExtendedRealityExperience) public extendedRealityExperiences;

    // Struct to represent an extended reality experience
    struct ExtendedRealityExperience {
        string experienceType;
        string experienceData;
    }

    // Event emitted when a new extended reality experience is created
    event ExtendedRealityExperienceCreatedEvent(address indexed user, ExtendedRealityExperience experience);

    // Function to create a new extended reality experience
    function createExtendedRealityExperience(string memory experienceType, string memory experienceData) public {
        ExtendedRealityExperience storage experience = extendedRealityExperiences[msg.sender];
        experience.experienceType = experienceType;
        experience.experienceData = experienceData;
        emit ExtendedRealityExperienceCreatedEvent(msg.sender, experience);
   }

    // Function to get an extended reality experience
    function getExtendedRealityExperience(address user) public view returns (ExtendedRealityExperience memory) {
        return extendedRealityExperiences[user];
    }
}
