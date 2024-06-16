pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract DecentralizedOracle {
    using SafeMath for uint256;

    // Mapping of data sources to their reputation scores
    mapping (address => uint256) public dataSources;

    // Mapping of data feeds to their aggregated values
    mapping (bytes32 => uint256) public dataFeeds;

    // Event emitted when a data source's reputation score is updated
    event DataSourceReputationUpdated(address dataSource, uint256 newScore);

    // Event emitted when a data feed is updated
    event DataFeedUpdated(bytes32 dataFeed, uint256 newValue);

    // Struct to represent a data source
    struct DataSource {
        uint256 reputationScore; // Reputation score of the data source
        uint256 numReports; // Number of reports submitted by the data source
    }

    // Function to update a data source's reputation score
    function updateDataSourceReputation(address dataSource, uint256 newScore) public {
        dataSources[dataSource] = newScore;
        emit DataSourceReputationUpdated(dataSource, newScore);
    }

    //Function to submit a data report
    function submitDataReport(bytes32 dataFeed, uint256 value) public {
        DataSource storage dataSource = dataSources[msg.sender];
        require(dataSource.reputationScore >= 500, "Reputation score is too low"); // 500 is the minimum reputation score required to submit a report
        dataSource.numReports++;
        dataFeeds[dataFeed] = calculateAggregatedValue(dataFeed, value);
        emit DataFeedUpdated(dataFeed, dataFeeds[dataFeed]);
    }

    // Function to calculate the aggregated value of a data feed
    function calculateAggregatedValue(bytes32 dataFeed, uint256 value) internal returns (uint256) {
        // Implement a weighted average or median calculation based on the reputation scores of the data sources
        // For simplicity, this example uses a simple average
        uint256 sum = 0;
        uint256 count = 0;
        for (address dataSource in dataSources) {
            sum += value * dataSources[dataSource].reputationScore;
            count += dataSources[dataSource].reputationScore;
        }
        return sum / count;
    }
}
