pragma solidity ^0.8.0;

import "https://github.com/chainlink/chainlink-contracts/contracts/src/v0.8/ChainlinkClient.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Roles.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract ChainlinkOracle is ChainlinkClient {
    using Roles for address;
    using SafeMath for uint256;

    // Events
    event NewDataRequest(address indexed requester, bytes32 requestId, string dataDescription);
    event DataReceived(address indexed requester, bytes32 requestId, bytes data);
    event DataFailed(address indexed requester, bytes32 requestId, string error);

    // Structs
    struct DataRequest {
        address requester;
        bytes32 requestId;
        string dataDescription;
        uint256 timestamp;
        bool fulfilled;
        bool rejected;
    }

    // Mapping of data requests
    mapping (bytes32 => DataRequest) public dataRequests;

    // Mapping of data providers
    mapping (address => bool) public dataProviders;

    // Constructor
    constructor(address _link, address _oracle) public {
        setChainlinkToken(_link);
        setOracle(_oracle);
    }

    // Function to request data from a data provider
    function requestData(string memory _dataDescription) public {
        bytes32 requestId = keccak256(abi.encodePacked(_dataDescription, block.timestamp));
        DataRequest storage request = dataRequests[requestId];
        request.requester = msg.sender;
        request.requestId = requestId;
        request.dataDescription = _dataDescription;
        request.timestamp = block.timestamp;
        request.fulfilled = false;
        request.rejected = false;
        emit NewDataRequest(msg.sender, requestId, _dataDescription);
        sendChainlinkRequest(requestId, _dataDescription, this.fulfillDataRequest.selector);
    }

    // Function to fulfill a data request
    function fulfillDataRequest(bytes32 _requestId, bytes memory _data) public recordChainlinkFulfillment(_requestId) {
        DataRequest storage request = dataRequests[_requestId];
        require(request.requester!= address(0), "Invalid request ID");
        request.fulfilled = true;
        emit DataReceived(request.requester, _requestId, _data);
    }

    // Function to reject a data request
    function rejectDataRequest(bytes32 _requestId, string memory _error) public {
        DataRequest storage request = dataRequests[_requestId];
        require(request.requester!= address(0), "Invalid request ID");
        request.rejected = true;
        emit DataFailed(request.requester, _requestId, _error);
    }

    // Modifier to restrict access to only the oracle
    modifier onlyOracle {
        require(msg.sender == oracle, "Only the oracle can fulfill or reject data requests");
        _;
    }

    // Modifier to restrict access to only data providers
    modifier onlyDataProvider {
        require(dataProviders[msg.sender], "Only data providers can fulfill data requests");
        _;
    }
}
