// evm.sol
pragma solidity ^0.8.0;

contract SmartContract {
    mapping (string => string) public storage;

    function store(string memory key, string memory value) public {
        storage[key] = value;
    }

    function retrieve(string memory key) public view returns (string memory) {
        return storage[key];
    }
}
