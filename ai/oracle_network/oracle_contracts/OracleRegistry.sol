pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Roles.sol";

contract OracleRegistry {
    using Roles for address;

    // Mapping of oracle providers to their respective APIs
    mapping(address => mapping(string => address)) public oracleProviders;

    // Event emitted when a new oracle provider is registered
    event OracleProviderRegistered(address indexed provider, string api);

    // Event emitted when an oracle provider is updated
    event OracleProviderUpdated(address indexed provider, string api);

    // Only allow the owner to register new oracle providers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can register oracle providers");
        _;
    }

    // Register a new oracle provider
    function registerOracleProvider(address provider, string memory api) public onlyOwner {
        oracleProviders[provider][api] = provider;
        emit OracleProviderRegistered(provider, api);
    }

    // Update an existing oracle provider
    function updateOracleProvider(address provider, string memory api) public onlyOwner {
        oracleProviders[provider][api] = provider;
        emit OracleProviderUpdated(provider, api);
    }
}
