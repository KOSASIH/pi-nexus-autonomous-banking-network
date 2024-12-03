// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/Counters.sol";
import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";

contract IdentityVerification is AccessControl, ERC721URIStorage {
    using Counters for Counters.Counter;
    Counters.Counter private _identityIdCounter;

    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant VERIFIER_ROLE = keccak256("VERIFIER_ROLE");

    struct Identity {
        uint256 id;
        address owner;
        string ipfsHash; // IPFS hash for off-chain data
        bool isVerified;
        uint256 verificationTimestamp;
    }

    mapping(uint256 => Identity) private _identities;
    mapping(address => uint256) private _ownerToIdentityId;

    event IdentityCreated(uint256 indexed identityId, address indexed owner, string ipfsHash);
    event IdentityVerified(uint256 indexed identityId, address indexed verifier);
    event IdentityRevoked(uint256 indexed identityId, address indexed admin);

    constructor() ERC721("IdentityToken", "IDT") {
        _setupRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _setupRole(VERIFIER_ROLE, msg.sender);
    }

    // Function to create a new identity
    function createIdentity(string memory ipfsHash) public {
        require(_ownerToIdentityId[msg.sender] == 0, "Identity already exists for this address");

        uint256 identityId = _identityIdCounter.current();
        _identities[identityId] = Identity(identityId, msg.sender, ipfsHash, false, 0);
        _ownerToIdentityId[msg.sender] = identityId;

        _mint(msg.sender, identityId);
        _setTokenURI(identityId, ipfsHash);

        emit IdentityCreated(identityId, msg.sender, ipfsHash);
        _identityIdCounter.increment();
    }

    // Function to verify an identity
    function verifyIdentity(uint256 identityId) public onlyRole(VERIFIER_ROLE) {
        require(_exists(identityId), "Identity does not exist");
        require(!_identities[identityId].isVerified, "Identity already verified");

        _identities[identityId].isVerified = true;
        _identities[identityId].verificationTimestamp = block.timestamp;

        emit IdentityVerified(identityId, msg.sender);
    }

    // Function to revoke an identity
    function revokeIdentity(uint256 identityId) public onlyRole(ADMIN_ROLE) {
        require(_exists(identityId), "Identity does not exist");
        require(_identities[identityId].isVerified, "Identity is not verified");

        _identities[identityId].isVerified = false;

        emit IdentityRevoked(identityId, msg.sender);
    }

    // Function to get identity details
    function getIdentityDetails(uint256 identityId) public view returns (Identity memory) {
        require(_exists(identityId), "Identity does not exist");
        return _identities[identityId];
    }

    // Function to get the identity ID of an address
    function getIdentityIdByOwner(address owner) public view returns (uint256) {
        return _ownerToIdentityId[owner];
    }

    // Override supportsInterface to include AccessControl
    function supportsInterface(bytes4 interfaceId) public view virtual override(ERC721, AccessControl) returns (bool) {
        return super.supportsInterface(interfaceId);
    }
}
