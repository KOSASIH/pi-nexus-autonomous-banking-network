pragma solidity ^0.8.0;

import "./OracleRegistry.sol";
import "./OracleProvider.sol";

contract OracleConsumer {
    // Oracle registry contract
    OracleRegistry public oracleRegistry;

    // Oracle provider contract
    OracleProvider public oracleProvider;

    // Mapping of API endpoints to their respective oracle providers
    mapping(string => address) public oracleProviders;

    // Event emitted when a new oracle provider is selected
    event OracleProviderSelected(string indexed api, address provider);

    // Constructor
    constructor(address oracleRegistryAddress, address oracleProviderAddress) public {
        oracleRegistry = OracleRegistry(oracleRegistryAddress);
        oracleProvider = OracleProvider(oracleProviderAddress);
    }

    // Select an oracle provider for a specific API
    function selectOracleProvider(string memory api) public {
        address provider = oracleRegistry.oracleProviders[msg.sender][api];
        require(provider != address(0), "Oracle provider not found");
        oracleProviders[api] = provider;
        emit OracleProviderSelected(api, provider);
    }

    // Get data from an oracle provider
    function getData(string memory api) public view returns (bytes32) {
        address provider = oracleProviders[api];
        require(provider != address(0), "Oracle provider not selected");
        // Call the oracle provider's API implementation
        bytes32 data = OracleProvider(provider).getAPIImplementation(api).getData();
        return data;
    }
}
