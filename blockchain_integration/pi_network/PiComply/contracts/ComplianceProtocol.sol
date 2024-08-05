pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Roles.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/chainlink/chainlink-contracts/contracts/src/v0.8/ChainlinkClient.sol";

contract ComplianceProtocol {
    using Roles for address;
    using SafeMath for uint256;

    // Events
    event NewComplianceRequest(address indexed requester, uint256 requestId);
    event ComplianceRequestFulfilled(address indexed requester, uint256 requestId, bool complianceStatus);
    event ComplianceRequestRejected(address indexed requester, uint256 requestId, string reason);

    // Structs
    struct ComplianceRequest {
        address requester;
        uint256 requestId;
        string regulatoryRequirement;
        uint256 timestamp;
        bool fulfilled;
        bool rejected;
    }

    // Mapping of compliance requests
    mapping (uint256 => ComplianceRequest) public complianceRequests;

    // Mapping of regulatory requirements to their corresponding compliance statuses
    mapping (string => bool) public regulatoryCompliance;

    // Chainlink oracle address
    address public oracleAddress;

    // Constructor
    constructor(address _oracleAddress) public {
        oracleAddress = _oracleAddress;
    }

    // Function to request compliance check
    function requestComplianceCheck(string memory _regulatoryRequirement) public {
        uint256 requestId = uint256(keccak256(abi.encodePacked(_regulatoryRequirement, block.timestamp)));
        ComplianceRequest storage request = complianceRequests[requestId];
        request.requester = msg.sender;
        request.requestId = requestId;
        request.regulatoryRequirement = _regulatoryRequirement;
        request.timestamp = block.timestamp;
        request.fulfilled = false;
        request.rejected = false;
        emit NewComplianceRequest(msg.sender, requestId);
    }

    // Function to fulfill compliance request
    function fulfillComplianceRequest(uint256 _requestId, bool _complianceStatus) public onlyOracle {
        ComplianceRequest storage request = complianceRequests[_requestId];
        require(request.requester != address(0), "Invalid request ID");
        request.fulfilled = true;
        regulatoryCompliance[request.regulatoryRequirement] = _complianceStatus;
        emit ComplianceRequestFulfilled(request.requester, _requestId, _complianceStatus);
    }

    // Function to reject compliance request
    function rejectComplianceRequest(uint256 _requestId, string memory _reason) public onlyOracle {
        ComplianceRequest storage request = complianceRequests[_requestId];
        require(request.requester != address(0), "Invalid request ID");
        request.rejected = true;
        emit ComplianceRequestRejected(request.requester, _requestId, _reason);
    }

    // Modifier to restrict access to only the oracle
    modifier onlyOracle {
        require(msg.sender == oracleAddress, "Only the oracle can fulfill or reject compliance requests");
        _;
    }
}
