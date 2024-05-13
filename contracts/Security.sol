pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

contract Security is Ownable, Pausable {
    // Access control variables
    mapping(address => bool) public isAdmin;

    // Event for when an address is granted admin privileges
    event AddressGrantedAdmin(address indexed address);

    // Event for when an address is revoked admin privileges
    event AddressRevokedAdmin(address indexed address);

    // Constructor
    constructor() {
        isAdmin[msg.sender] = true;
    }

    // Function to grant admin privileges to an address
    function grantAdmin(address address) public onlyOwner {
        require(!isAdmin[address], "Address already has admin privileges");

        isAdmin[address] = true;

        // Emit an event for when an address is granted admin privileges
        emit AddressGrantedAdmin(address);
    }

    // Function to revoke admin privileges from an address
    function revokeAdmin(address address) public onlyOwner {
        require(isAdmin[address], "Address does not have admin privileges");

        isAdmin[address] = false;

        // Emit an event for when an address is revoked admin privileges
        emit AddressRevokedAdmin(address);
    }

    // Function to pause the contract
    function pause() public onlyOwner {
        _pause();
    }

    // Function to unpause the contract
    function unpause() public onlyOwner {
        _unpause();
    }

    // Function to check if an address has admin privileges
    function isAdmin(address address) public view returns (bool) {
        return isAdmin[address];
    }
}
