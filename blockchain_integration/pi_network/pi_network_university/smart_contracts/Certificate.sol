pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC721/ERC721.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Roles.sol";
import "./CertificateVerification.sol";

contract Certificate is ERC721 {
    using Roles for address;

    // Mapping of certificate IDs to their corresponding metadata
    mapping (uint256 => CertificateMetadata) public certificateMetadata;

    // Mapping of certificate IDs to their corresponding verification status
    mapping (uint256 => bool) public certificateVerifications;

    // Event emitted when a certificate is minted
    event CertificateMinted(uint256 certificateId, address owner);

    // Event emitted when a certificate is transferred
    event CertificateTransferred(uint256 certificateId, address from, address to);

    // Event emitted when a certificate is verified
    event CertificateVerified(uint256 certificateId, bool isValid);

    // Role for certificate owners
    bytes32 public constant OWNER_ROLE = keccak256("OWNER_ROLE");

    // Role for certificate issuers
    bytes32 public constant ISSUER_ROLE = keccak256("ISSUER_ROLE");

    // Modifier to restrict access to certificate owners
    modifier onlyOwner(uint256 certificateId) {
        require(hasRole(OWNER_ROLE, msg.sender) && ownerOf(certificateId) == msg.sender, "Only owners can perform this action");
        _;
    }

    // Modifier to restrict access to certificate issuers
    modifier onlyIssuer() {
        require(hasRole(ISSUER_ROLE, msg.sender), "Only issuers can perform this action");
        _;
    }

    // Struct to represent certificate metadata
    struct CertificateMetadata {
        string name;
        string description;
        uint256 issuanceDate;
        uint256 expirationDate;
    }

    // Function to mint a new certificate
    function mintCertificate(address owner, string memory name, string memory description, uint256 issuanceDate, uint256 expirationDate) public onlyIssuer {
        uint256 certificateId = totalSupply();
        _mint(owner, certificateId);
        certificateMetadata[certificateId] = CertificateMetadata(name, description, issuanceDate, expirationDate);
        emit CertificateMinted(certificateId, owner);
    }

        // Function to transfer a certificate
    function transferCertificate(uint256 certificateId, address to) public onlyOwner(certificateId) {
        _transfer(from, to, certificateId);
        emit CertificateTransferred(certificateId, from, to);
    }

    // Function to verify a certificate
    function verifyCertificate(uint256 certificateId, bool isValid) public {
        certificateVerifications[certificateId] = isValid;
        emit CertificateVerified(certificateId, isValid);
    }

    // Function to get the metadata of a certificate
    function getCertificateMetadata(uint256 certificateId) public view returns (CertificateMetadata memory) {
        return certificateMetadata[certificateId];
    }

    // Function to get the verification status of a certificate
    function getCertificateVerification(uint256 certificateId) public view returns (bool) {
        return certificateVerifications[certificateId];
    }
}
