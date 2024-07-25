// KYC.sol
pragma solidity ^0.8.10;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Roles.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Counters.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Strings.sol";

contract KYC {
    using Roles for address;
    using Counters for Counters.Counter;
    using Strings for string;

    // Mapping of user addresses to their KYC status
    mapping (address => bool) public kycStatus;

    // Mapping of user addresses to their KYC verification data
    mapping (address => string) public kycVerificationData;

    // Mapping of user addresses to their KYC document hashes
    mapping (address => bytes32[]) public kycDocumentHashes;

    // Event emitted when a user's KYC status is updated
    event KYCStatusUpdated(address indexed user, bool status);

    // Event emitted when a user's KYC verification data is updated
    event KYCVerificationDataUpdated(address indexed user, string data);

    // Event emitted when a user's KYC document hashes are updated
    event KYCDocumentHashesUpdated(address indexed user, bytes32[] hashes);

    // Only allow the KYC administrator to update KYC status
    modifier onlyKYCAdmin {
        require(msg.sender.hasRole("KYC_ADMIN"), "Only KYC admin can update KYC status");
        _;
    }

    // Function to update a user's KYC status
    function updateKYCStatus(address user, bool status) public onlyKYCAdmin {
        kycStatus[user] = status;
        emit KYCStatusUpdated(user, status);
    }

    // Function to update a user's KYC verification data
    function updateKYCVerificationData(address user, string memory data) public onlyKYCAdmin {
        kycVerificationData[user] = data;
        emit KYCVerificationDataUpdated(user, data);
    }

    // Function to update a user's KYC document hashes
    function updateKYCDocumentHashes(address user, bytes32[] memory hashes) public onlyKYCAdmin {
        kycDocumentHashes[user] = hashes;
        emit KYCDocumentHashesUpdated(user, hashes);
    }

    // Function to check a user's KYC status
    function getKYCStatus(address user) public view returns (bool) {
        return kycStatus[user];
    }

    // Function to check a user's KYC verification data
    function getKYCVerificationData(address user) public view returns (string memory) {
        return kycVerificationData[user];
    }

    // Function to check a user's KYC document hashes
    function getKYCDocumentHashes(address user) public view returns (bytes32[] memory) {
        return kycDocumentHashes[user];
    }

    // Function to verify a user's KYC document hash
    function verifyKYCDocumentHash(address user, bytes32 hash) public view returns (bool) {
        for (uint256 i = 0; i < kycDocumentHashes[user].length; i++) {
            if (kycDocumentHashes[user][i] == hash) {
                return true;
            }
        }
        return false;
    }
}
