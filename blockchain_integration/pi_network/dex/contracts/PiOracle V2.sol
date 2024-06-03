pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";
import "https://github.com/smartcontractkit/chainlink/blob/master/evm-contracts/src/v0.6/ChainlinkClient.sol";

contract PiOracleV2 {
    using SafeMath for uint256;

    // Mapping of data sources
    mapping (address => DataSource) public dataSources;

    // Event emitted when new data is received
    event NewData(address indexed dataSource, uint256 data);

    // Struct to represent a data source
    struct DataSource {
        address dataSource;
        uint256 data;
        uint256 timestamp;
    }

    // Function to add a new data source
    function addDataSource(address dataSource) public {
        // Create a new data source
        DataSource memory source = DataSource(dataSource, 0, 0);
        dataSources[dataSource] = source;
    }

    // Function to request data from a data source
    function requestData(address dataSource) public {
        // Create a new Chainlink request
        Chainlink.Request memory req = new Chainlink.Request(dataSource, "get_data", "");

        // Send the request and get the response
        uint256 data = req.execute();

        // Update the data source with the new data
        dataSources[dataSource].data = data;
        dataSources[dataSource].timestamp = block.timestamp;

        // Emit the NewData event
        emit NewData(dataSource, data);
    }

    // Function to get the latest data from a data source
    function getLatestData(address dataSource) public view returns (uint256) {
        return dataSources[dataSource].data;
    }

    // Function to get the latest timestamp from a data source
    function getLatestTimestamp(address dataSource) public view returns (uint256) {
        return dataSources[dataSource].timestamp;
    }
}
