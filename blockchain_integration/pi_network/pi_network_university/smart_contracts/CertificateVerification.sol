pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC721/SafeERC721.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Roles.sol";

contract CertificateVerification {
    using SafeERC721 for address;
    using Roles for address;

    // Mapping of certificate hashes to their corresponding verification status
    mapping (bytes32 => bool) public certificateVerifications;

    // Mapping of certificate hashes to their corresponding issuers
    mapping (bytes32 => address) public certificateIssuers;

    // Event emitted when a certificate is verified
    event CertificateVerified(bytes32 certificateHash, bool isValid);

    // Event emitted when a certificate is revoked
    event CertificateRevoked(bytes32 certificateHash);

    // Role for certificate issuers
    bytes32 public constant ISSUER_ROLE = keccak256("ISSUER_ROLE");

    // Role for certificate verifiers
    bytes32 public constant VERIFIER_ROLE = keccak256("VERIFIER_ROLE");

    // Modifier to restrict access to certificate issuers
    modifier onlyIssuer() {
        require(hasRole(ISSUER_ROLE, msg.sender), "Only issuers can perform this action");
        _;
    }

    // Modifier to restrict access to certificate verifiers
    modifier onlyVerifier() {
        require(hasRole(VERIFIER_ROLE, msg.sender), "Only verifiers can perform this action");
        _;
    }

    // Function to verify a certificate
    function verifyCertificate(bytes32 certificateHash, bool isValid) public onlyVerifier {
        certificateVerifications[certificateHash] = isValid;
        emit CertificateVerified(certificateHash, isValid);
    }

    // Function to revoke a certificate
    function revokeCertificate(bytes32 certificateHash) public onlyIssuer {
        certificateVerifications[certificateHash] = false;
        emit CertificateRevoked(certificateHash);
    }

    // Function to issue a new certificate
    function issueCertificate(bytes32 certificateHash, address issuer) public onlyIssuer {
        certificateIssuers[certificateHash] = issuer;
    }

    // Function to get the verification status of a certificate
    function getCertificateVerification(bytes32 certificateHash) public view returns (bool) {
        return certificateVerifications[certificateHash];
    }

    // Function to get the issuer of a certificate
    function getCertificateIssuer(bytes32 certificateHash) public view returns (address) {
        return certificateIssuers[certificateHash];
    }
}
