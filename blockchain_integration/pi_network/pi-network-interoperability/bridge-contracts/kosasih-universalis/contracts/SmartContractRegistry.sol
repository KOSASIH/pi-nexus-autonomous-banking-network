pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Address.sol";

contract SmartContractRegistry {
    mapping(address => bytes) public contractABIs;

    constructor() public {
        // Initialize the contract registry
    }

    function registerContract(address _contractAddress, bytes _contractABI) public {
        // Register a new contract ABI
        contractABIs[_contractAddress] = _contractABI;
    }

    function getContractABI(address _contractAddress) public view returns (bytes) {
        // Retrieve a contract ABI
        return contractABIs[_contractAddress];
    }
}
