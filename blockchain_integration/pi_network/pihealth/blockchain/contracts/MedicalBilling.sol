pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/AccessControl.sol";

contract MedicalBilling is SafeERC20, AccessControl {
    // Mapping of patient IDs to their medical billing information
    mapping(address => mapping(uint256 => MedicalBillingData)) public patientBilling;

    // Mapping of healthcare provider IDs to their billing permissions
    mapping(address => mapping(uint256 => bool)) public providerBillingPermissions;

    // Event emitted when a new medical billing record is created
    event NewMedicalBilling(address patient, uint256 billingId);

    // Event emitted when a healthcare provider is granted billing permissions
    event GrantBillingAccess(address provider, uint256 billingId);

    // Event emitted when a healthcare provider is revoked billing permissions
    event RevokeBillingAccess(address provider, uint256 billingId);

    // Struct to represent a medical billing record
    struct MedicalBillingData {
        uint256 billingId;
        string patientName;
        string medicalProcedure;
        uint256 cost;
        bool paid;
    }

    // Function to create a new medical billing record
    function createMedicalBilling(address patient, string memory patientName, string memory medicalProcedure, uint256 cost) public {
        // Generate a unique billing ID
        uint256 billingId = uint256(keccak256(abi.encodePacked(patient, patientName)));

        // Create a new medical billing record
                patientBilling[patient][billingId] = MedicalBillingData(billingId, patientName, medicalProcedure, cost, false);

        // Emit an event to notify of the new medical billing record
        emit NewMedicalBilling(patient, billingId);
    }

    // Function to grant billing permissions to a healthcare provider
    function grantBillingAccess(address provider, uint256 billingId) public {
        // Check if the provider is already granted billing permissions
        require(!providerBillingPermissions[provider][billingId], "Provider already has billing permissions");

        // Grant billing permissions to the provider
        providerBillingPermissions[provider][billingId] = true;

        // Emit an event to notify of the granted billing permissions
        emit GrantBillingAccess(provider, billingId);
    }

    // Function to revoke billing permissions from a healthcare provider
    function revokeBillingAccess(address provider, uint256 billingId) public {
        // Check if the provider is already revoked billing permissions
        require(providerBillingPermissions[provider][billingId], "Provider does not have billing permissions");

        // Revoke billing permissions from the provider
        providerBillingPermissions[provider][billingId] = false;

        // Emit an event to notify of the revoked billing permissions
        emit RevokeBillingAccess(provider, billingId);
    }

    // Function to get a patient's medical billing record
    function getMedicalBilling(address patient, uint256 billingId) public view returns (MedicalBillingData memory) {
        // Check if the patient has a medical billing record with the given ID
        require(patientBilling[patient][billingId].billingId != 0, "Medical billing record not found");

        // Return the patient's medical billing record
        return patientBilling[patient][billingId];
    }

    // Function to pay a medical billing record
    function payMedicalBilling(address patient, uint256 billingId) public {
        // Check if the patient has a medical billing record with the given ID
        require(patientBilling[patient][billingId].billingId != 0, "Medical billing record not found");

        // Check if the billing record is already paid
        require(!patientBilling[patient][billingId].paid, "Billing record is already paid");

        // Mark the billing record as paid
        patientBilling[patient][billingId].paid = true;
    }
}
