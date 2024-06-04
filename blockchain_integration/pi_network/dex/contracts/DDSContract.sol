pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";

contract DDSContract is Ownable {
    struct File {
        address owner;
        string cid;
        string name;
        string description;
        uint256 createdAt;
    }

    mapping(address => File[]) public files;

    event FileUploaded(address indexed owner, string cid, string name, string description, uint256 createdAt);
    event FileUpdated(address indexed owner, string cid, string name, string description, uint256 updatedAt);

    function uploadFile(string memory cid, string memory name, string memory description) external {
        //...
    }

    function updateFile(string memory cid, string memory name, string memory description) external {
        //...
    }

    function getFile(address owner, uint256 index) public view returns (File memory) {
        //...
    }
}
