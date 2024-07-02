pragma solidity ^0.8.0;

contract DecentralizedDataStorage {
    mapping (address => mapping (string => bytes)) public data;

    constructor() {
        // Initialize data storage
    }

    function storeData(string memory key, bytes memory value) public {
        data[msg.sender][key] = value;
    }

    function retrieveData(string memory key) public view returns (bytes memory) {
        return data[msg.sender][key];
    }
}
