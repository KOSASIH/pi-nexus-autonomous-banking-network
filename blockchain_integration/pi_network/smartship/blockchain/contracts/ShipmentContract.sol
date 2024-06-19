pragma solidity ^0.8.10;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC721/SafeERC721.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Counters.sol";

contract ShipmentContract {
    using SafeERC721 for address;
    using SafeMath for uint256;
    using Counters for Counters.Counter;

    // Mapping of shipment IDs to their respective shipment data
    mapping(uint256 => Shipment) public shipments;

    // Mapping of shipment IDs to their respective tracking data
    mapping(uint256 => TrackingData[]) public shipmentTrackingData;

    // Mapping of shipment IDs to their respective statuses
    mapping(uint256 => ShipmentStatus) public shipmentStatuses;

    // Event emitted when a new shipment is created
    event NewShipment(uint256 indexed shipmentId, address indexed logisticsProvider);

    // Event emitted when a shipment status is updated
    event ShipmentStatusUpdate(uint256 indexed shipmentId, ShipmentStatus status);

    // Event emitted when a new tracking data is added to a shipment
    event TrackingDataAdded(uint256 indexed shipmentId, uint256 indexed trackingDataIndex, uint256 timestamp, string location, string status);

    // Enum for shipment status
    enum ShipmentStatus {
        CREATED,
        IN_TRANSIT,
        DELIVERED,
        LOST,
        DAMAGED
    }

    // Struct for shipment data
    struct Shipment {
        uint256 id;
        address logisticsProvider;
        uint256 createdAt;
        uint256 updatedAt;
        ShipmentStatus status;
    }

    // Struct for tracking data
    struct TrackingData {
        uint256 timestamp;
        string location;
        string status;
    }

    // Counter for shipment IDs
    Counters.Counter public shipmentIdCounter;

    // Modifier to check if the caller is the logistics provider for the shipment
    modifier onlyLogisticsProvider(uint256 _shipmentId) {
        require(shipments[_shipmentId].logisticsProvider == msg.sender, "Only the logistics provider assigned to the shipment can call this function");
        _;
    }

    // Function to create a new shipment
    function createShipment(address _logisticsProvider) public {
        shipmentIdCounter.increment();
        uint256 shipmentId = shipmentIdCounter.current();
        shipments[shipmentId] = Shipment(shipmentId, _logisticsProvider, block.timestamp, block.timestamp, ShipmentStatus.CREATED);
        emit NewShipment(shipmentId, _logisticsProvider);
    }

    // Function to update a shipment status
    function updateShipmentStatus(uint256 _shipmentId, ShipmentStatus _status) public onlyLogisticsProvider(_shipmentId) {
        shipmentStatuses[_shipmentId] = _status;
        shipments[_shipmentId].updatedAt = block.timestamp;
        emit ShipmentStatusUpdate(_shipmentId, _status);
    }

    // Function to add tracking data to a shipment
    function addTrackingData(uint256 _shipmentId, uint256 _timestamp, string memory _location, string memory _status) public onlyLogisticsProvider(_shipmentId) {
        TrackingData memory trackingData = TrackingData(_timestamp, _location, _status);
        shipmentTrackingData[_shipmentId].push(trackingData);
        emit TrackingDataAdded(_shipmentId, shipmentTrackingData[_shipmentId].length - 1, _timestamp, _location, _status);
    }

    // Function to get a shipment by ID
    function getShipment(uint256 _shipmentId) public view returns (Shipment memory) {
        return shipments[_shipmentId];
    }

    // Function to get all tracking data for a shipment
    function getTrackingDataForShipment(uint256 _shipmentId) public view returns (TrackingData[] memory) {
        return shipmentTrackingData[_shipmentId];
    }

    // Function to get the status of a shipment
    function getShipmentStatus(uint256 _shipmentId) public view returns (ShipmentStatus) {
        return shipmentStatuses[_shipmentId];
    }
}
