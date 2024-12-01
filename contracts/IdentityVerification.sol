// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract IdentityVerification {
    struct Identity {
        string name;
        string email;
        string phoneNumber;
        bool isVerified;
        address verifier;
    }

    mapping(address => Identity) private identities;
    mapping(address => bool) private verifiers;

    event IdentityRegistered(address indexed user, string name, string email, string phoneNumber);
    event IdentityVerified(address indexed user, address indexed verifier);
    event IdentityUpdated(address indexed user, string name, string email, string phoneNumber);

    modifier onlyVerifier() {
        require(verifiers[msg.sender], "Not an authorized verifier");
        _;
    }

    modifier identityExists(address user) {
        require(bytes(identities[user].name).length > 0, "Identity does not exist");
        _;
    }

    // Function to register a new identity
    function registerIdentity(string memory _name, string memory _email, string memory _phoneNumber) public {
        require(bytes(identities[msg.sender].name).length == 0, "Identity already registered");

        identities[msg.sender] = Identity({
            name: _name,
            email: _email,
            phoneNumber: _phoneNumber,
            isVerified: false,
            verifier: address(0)
        });

        emit IdentityRegistered(msg.sender, _name, _email, _phoneNumber);
    }

    // Function to verify an identity
    function verifyIdentity(address _user) public onlyVerifier identityExists(_user) {
        identities[_user].isVerified = true;
        identities[_user].verifier = msg.sender;

        emit IdentityVerified(_user, msg.sender);
    }

    // Function to update identity information
    function updateIdentity(string memory _name, string memory _email, string memory _phoneNumber) public identityExists(msg.sender) {
        identities[msg.sender].name = _name;
        identities[msg.sender].email = _email;
        identities[msg.sender].phoneNumber = _phoneNumber;

        emit IdentityUpdated(msg.sender, _name, _email, _phoneNumber);
    }

    // Function to get identity information
    function getIdentity(address _user) public view returns (string memory, string memory, string memory, bool, address) {
        Identity memory identity = identities[_user];
        return (identity.name, identity.email, identity.phoneNumber, identity.isVerified, identity.verifier);
    }

    // Function to add a verifier
    function addVerifier(address _verifier) public {
        require(!verifiers[_verifier], "Already a verifier");
        verifiers[_verifier] = true;
    }

    // Function to remove a verifier
    function removeVerifier(address _verifier) public {
        require(verifiers[_verifier], "Not a verifier");
        verifiers[_verifier] = false;
    }
}
