// erc725_contract.sol
pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC725/SafeERC725.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC725/ERC725.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Roles.sol";

contract ERC725Contract is SafeERC725, ERC725, Roles {
    // Mapping of user IDs to their corresponding identity data
    mapping (address => Identity) public identities;

    // Event emitted when a new identity is created
    event NewIdentity(address indexed userId, bytes32 identityId);

    // Event emitted when an identity is updated
    event UpdateIdentity(address indexed userId, bytes32 identityId);

    // Event emitted when an identity is deleted
    event DeleteIdentity(address indexed userId, bytes32 identityId);

    // Struct to represent an identity
    struct Identity {
        bytes32 id;
        string name;
        string email;
        string phoneNumber;
        bytes32[] credentials;
    }

    // Modifier to check if the caller is the owner of the identity
    modifier onlyOwner(address userId) {
        require(msg.sender == userId, "Only the owner can perform this action");
        _;
    }

    // Function to create a new identity
    function createIdentity(string memory _name, string memory _email, string memory _phoneNumber) public {
        bytes32 identityId = keccak256(abi.encodePacked(_name, _email, _phoneNumber));
        Identity storage identity = identities[msg.sender];
        identity.id = identityId;
        identity.name = _name;
        identity.email = _email;
        identity.phoneNumber = _phoneNumber;
        emit NewIdentity(msg.sender, identityId);
    }

    // Function to update an identity
    function updateIdentity(string memory _name, string memory _email, string memory _phoneNumber) public onlyOwner(msg.sender) {
        Identity storage identity = identities[msg.sender];
        identity.name = _name;
        identity.email = _email;
        identity.phoneNumber = _phoneNumber;
        emit UpdateIdentity(msg.sender, identity.id);
    }

    // Function to delete an identity
    function deleteIdentity() public onlyOwner(msg.sender) {
        delete identities[msg.sender];
        emit DeleteIdentity(msg.sender, identities[msg.sender].id);
    }

    // Function to add a credential to an identity
    function addCredential(bytes32 _credential) public onlyOwner(msg.sender) {
        Identity storage identity = identities[msg.sender];
        identity.credentials.push(_credential);
    }

    // Function to remove a credential from an identity
    function removeCredential(bytes32 _credential) public onlyOwner(msg.sender) {
        Identity storage identity = identities[msg.sender];
        for (uint256 i = 0; i < identity.credentials.length; i++) {
            if (identity.credentials[i] == _credential) {
                identity.credentials[i] = identity.credentials[identity.credentials.length - 1];
                identity.credentials.pop();
                break;
            }
        }
    }

    // Function to get an identity by ID
    function getIdentity(bytes32 _identityId) public view returns (Identity memory) {
        return identities[_identityId];
    }

    // Function to get all identities
    function getAllIdentities() public view returns (Identity[] memory) {
        Identity[] memory identitiesArray = new Identity[](identities.length);
        for (uint256 i = 0; i < identities.length; i++) {
            identitiesArray[i] = identities[i];
        }
        return identitiesArray;
    }
}
