pragma solidity ^0.8.0;

contract DataFeed {
    // Mapping of data feed IDs to their respective data
    mapping (bytes32 => bytes) public dataFeeds;

    // Event emitted when a new data feed is created
    event NewDataFeed(bytes32 indexed dataFeedId, bytes data);

    // Event emitted when a data feed is updated
    event UpdatedDataFeed(bytes32 indexed dataFeedId, bytes data);

    // Event emitted when a data feed is removed
    event RemovedDataFeed(bytes32 indexed dataFeedId);

    // Function to create a new data feed
    function createDataFeed(bytes32 _dataFeedId, bytes _data) public {
        // Only allow authorized oracles to create data feeds
        require(OracleNexus(msg.sender).isAuthorizedOracle(msg.sender), "Only authorized oracles can create data feeds");

        // Create the data feed
        dataFeeds[_dataFeedId] = _data;

        // Emit the NewDataFeed event
        emit NewDataFeed(_dataFeedId, _data);
    }

    // Function to update a data feed
    function updateDataFeed(bytes32 _dataFeedId, bytes _data) public {
        // Only allow authorized oracles to update data feeds
        require(OracleNexus(msg.sender).isAuthorizedOracle(msg.sender), "Only authorized oracles can update data feeds");

        // Update the data feed
        dataFeeds[_dataFeedId] = _data;

        // Emit the UpdatedDataFeed event
        emit UpdatedDataFeed(_dataFeedId, _data);
    }

    // Function to remove a data feed
    function removeDataFeed(bytes32 _dataFeedId) public {
        // Only allow authorized oracles to remove data feeds
        require(OracleNexus(msg.sender).isAuthorizedOracle(msg.sender), "Only authorized oracles can remove data feeds");

        // Remove the data feed
        delete dataFeeds[_dataFeedId];

        // Emit the RemovedDataFeed event
        emit RemovedDataFeed(_dataFeedId);
    }

    // Function to get the data for a given data feed ID
    function getData(bytes32 _dataFeedId) public view returns (bytes) {
        return dataFeeds[_dataFeedId];
    }
}
