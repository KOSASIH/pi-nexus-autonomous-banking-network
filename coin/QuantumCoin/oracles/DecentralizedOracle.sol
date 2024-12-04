// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";

contract DecentralizedOracle is Ownable {
    struct DataFeed {
        string name;
        uint256 value;
        uint256 timestamp;
        address provider;
    }

    mapping(bytes32 => DataFeed) public dataFeeds;

    event DataFeedUpdated(bytes32 indexed feedId, uint256 value, address indexed provider);

    // Register a new data feed
    function registerDataFeed(bytes32 feedId, string memory name) external onlyOwner {
        require(dataFeeds[feedId].provider == address(0), "Data feed already exists");
        dataFeeds[feedId] = DataFeed(name, 0, block.timestamp, msg.sender);
    }

    // Update the value of a data feed
    function updateDataFeed(bytes32 feedId, uint256 value) external {
        require(dataFeeds[feedId].provider == msg.sender, "Only the provider can update the data feed");
        dataFeeds[feedId].value = value;
        dataFeeds[feedId].timestamp = block.timestamp;

        emit DataFeedUpdated(feedId, value, msg.sender);
    }

    // Retrieve the latest value of a data feed
    function getDataFeed(bytes32 feedId) external view returns (string memory name, uint256 value, uint256 timestamp, address provider) {
        DataFeed memory feed = dataFeeds[feedId];
        return (feed.name, feed.value, feed.timestamp, feed.provider);
    }
}
