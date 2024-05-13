pragma solidity ^0.8.0;

import "./OracleConsumer.sol";

contract OracleAggregator is OracleConsumer {
    // Aggregated data
    mapping(string => bytes32) public aggregatedData;

    // Event emitted when aggregated data is updated
    event AggregatedDataUpdated(string indexed api, bytes32 data);

    constructor(address oracleRegistryAddress, address oracleProviderAddress) OracleConsumer(oracleRegistryAddress, oracleProviderAddress) {}

    // Update aggregated data
    function updateAggregatedData(string memory api) public {
        bytes32 data = getData(api);
        aggregatedData[api] = data;
        emit AggregatedDataUpdated(api, data);
    }
}
