pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Roles.sol";

contract OracleProvider {
    using Roles for address;

    // Mapping of API endpoints to their respective implementations
    mapping(string => address) public apiImplementations;

    // Event emitted when a new API implementation is registered
    event APIImplementationRegistered(string indexed api, address implementation);

    // Event emitted when an API implementation is updated
    event APIImplementationUpdated(string indexed api, address implementation);

    // Only allow the owner to register new API implementations
    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can register API implementations");
        _;
    }

    // Register a new API implementation
    function registerAPIImplementation(string memory api, address implementation) public onlyOwner {
        apiImplementations[api] = implementation;
        emit APIImplementationRegistered(api, implementation);
    }

    // Update an existing API implementation
    function updateAPIImplementation(string memory api, address implementation) public onlyOwner {
        apiImplementations[api] = implementation;
        emit APIImplementationUpdated(api, implementation);
    }
}
