pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC721/SafeERC721.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Roles.sol";

contract ContractSphere {
    using SafeERC721 for address;
    using Roles for address;

    // Mapping of contract IDs to contract metadata
    mapping (uint256 => Contract) public contracts;

    // Mapping of user addresses to their owned contracts
    mapping (address => uint256[]) public userContracts;

    // Event emitted when a new contract is created
    event NewContract(uint256 contractId, address owner);

    // Event emitted when a contract is transferred
    event Transfer(uint256 contractId, address from, address to);

    // Event emitted when a contract is updated
    event Update(uint256 contractId, string metadata);

    // Struct to represent a contract
    struct Contract {
        uint256 id;
        string metadata;
        address owner;
    }

    // Modifier to restrict access to contract owners
    modifier onlyOwner(uint256 contractId) {
        require(msg.sender == contracts[contractId].owner, "Only the contract owner can perform this action");
        _;
    }

    // Function to create a new contract
    function createContract(string memory metadata) public {
        uint256 contractId = uint256(keccak256(abi.encodePacked(metadata)));
        contracts[contractId] = Contract(contractId, metadata, msg.sender);
        userContracts[msg.sender].push(contractId);
        emit NewContract(contractId, msg.sender);
    }

    // Function to transfer a contract
    function transferContract(uint256 contractId, address to) public onlyOwner(contractId) {
        contracts[contractId].owner = to;
        userContracts[to].push(contractId);
        userContracts[msg.sender].remove(contractId);
        emit Transfer(contractId, msg.sender, to);
    }

    // Function to update a contract
    function updateContract(uint256 contractId, string memory metadata) public onlyOwner(contractId) {
        contracts[contractId].metadata = metadata;
        emit Update(contractId, metadata);
    }

    // Function to get a contract by ID
    function getContract(uint256 contractId) public view returns (Contract memory) {
        return contracts[contractId];
    }

    // Function to get all contracts owned by a user
    function getUserContracts(address user) public view returns (uint256[] memory) {
        return userContracts[user];
    }
}
