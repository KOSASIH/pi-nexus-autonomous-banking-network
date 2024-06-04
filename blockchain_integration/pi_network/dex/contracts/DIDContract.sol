pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";

contract DIDContract is Ownable {
    struct DID {
        address owner;
        string name;
        string description;
        uint256 createdAt;
    }

    mapping(address => DID) public dids;

    event DIDCreated(address indexed owner, string name, string description, uint256 createdAt);
    event DIDUpdated(address indexed owner, string name, string description, uint256 updatedAt);

    function createDID(string memory name, string memory description) external {
        //...
    }

    function updateDID(string memory name, string memory description) external {
        //...
    }

    function getDID(address owner) public view returns (DID memory) {
        //...
    }
}
