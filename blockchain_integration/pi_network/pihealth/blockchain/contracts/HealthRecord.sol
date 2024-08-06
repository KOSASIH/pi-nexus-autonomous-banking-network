pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC721/SafeERC721.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/AccessControl.sol";

contract HealthRecord is SafeERC721, AccessControl {
    // Mapping of patient IDs to their health records
    mapping(address => mapping(uint256 => HealthRecordData)) public patientRecords;

    // Mapping of healthcare provider IDs to their access permissions
    mapping(address => mapping(uint256 => bool)) public providerPermissions;

    // Event emitted when a new health record is created
    event NewHealthRecord(address patient, uint256 recordId);

    // Event emitted when a healthcare provider is granted access to a health record
    event GrantAccess(address provider, uint256 recordId);

    // Event emitted when a healthcare provider is revoked access to a health record
    event RevokeAccess(address provider, uint256 recordId);

    // Struct to represent a health record
    struct HealthRecordData {
        uint256 recordId;
        string patientName;
        string medicalHistory;
        string allergies;
        string medications;
    }

    // Function to create a new health record
    function createHealthRecord(address patient, string memory patientName, string memory medicalHistory, string memory allergies, string memory medications) public {
        // Generate a unique record ID
        uint256 recordId = uint256(keccak256(abi.encodePacked(patient, patientName)));

        // Create a new health record
        patientRecords[patient][recordId] = HealthRecordData(recordId, patientName, medicalHistory, allergies, medications);

        // Emit an event to notify of the new health record
        emit NewHealthRecord(patient, recordId);
    }

    // Function to grant access to a healthcare provider
    function grantAccess(address provider, uint256 recordId) public {
        // Check if the provider is already granted access
        require(!providerPermissions[provider][recordId], "Provider already has access");

        // Grant access to the provider
        providerPermissions[provider][recordId] = true;

        // Emit an event to notify of the granted access
        emit GrantAccess(provider, recordId);
    }

    // Function to revoke access from a healthcare provider
    function revokeAccess(address provider, uint256 recordId) public {
        // Check if the provider is already revoked access
        require(providerPermissions[provider][recordId], "Provider does not have access");

        // Revoke access from the provider
        providerPermissions[provider][recordId] = false;

        // Emit an event to notify of the revoked access
        emit RevokeAccess(provider, recordId);
    }

    // Function to get a patient's health record
    function getHealthRecord(address patient, uint256 recordId) public view returns (HealthRecordData memory) {
        // Check if the patient has a health record with the given ID
        require(patientRecords[patient][recordId].recordId != 0, "Health record not found");

        // Return the patient's health record
        return patientRecords[patient][recordId];
    }
}
