pragma solidity ^0.8.10;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC721/SafeERC721.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Counters.sol";

contract LogisticsContract {
    using SafeERC721 for address;
    using SafeMath for uint256;
    using Counters for Counters.Counter;

    // Mapping of logistics providers to their respective ratings
    mapping(address => uint256) public logisticsProviders;

    // Mapping of shipment IDs to their respective logistics providers
    mapping(uint256 => address) public shipmentLogisticsProviders;

    // Mapping of shipment IDs to their respective statuses
    mapping(uint256 => ShipmentStatus) public shipmentStatuses;

    // Event emitted when a new logistics provider is registered
    event NewLogisticsProvider(address indexed provider, uint256 rating);

    // Event emitted when a shipment is created
    event NewShipment(uint256 indexed shipmentId, address indexed logisticsProvider);

    // Event emitted when a shipment status is updated
    event ShipmentStatusUpdate(uint256 indexed shipmentId, ShipmentStatus status);

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

    // Array of shipments
    Shipment[] public shipments;

    // Counter for shipment IDs
    Counters.Counter public shipmentIdCounter;

    // Modifier to check if the caller is a registered logistics provider
    modifier onlyLogisticsProvider() {
        require(logisticsProviders[msg.sender] > 0, "Only registered logistics providers can call this function");
        _;
    }

    // Function to register a new logistics provider
    function registerLogisticsProvider(uint256 _rating) public {
        logisticsProviders[msg.sender] = _rating;
        emit NewLogisticsProvider(msg.sender, _rating);
    }

    // Function to create a new shipment
    function createShipment(address _logisticsProvider) public onlyLogisticsProvider {
        shipmentIdCounter.increment();
        uint256 shipmentId = shipmentIdCounter.current();
        shipments.push(Shipment(shipmentId, _logisticsProvider, block.timestamp, block.timestamp, ShipmentStatus.CREATED));
        shipmentLogisticsProviders[shipmentId] = _logisticsProvider;
        shipmentStatuses[shipmentId] = ShipmentStatus.CREATED;
        emit NewShipment(shipmentId, _logisticsProvider);
    }

    // Function to update a shipment status
    function updateShipmentStatus(uint256 _shipmentId, ShipmentStatus _status) public onlyLogisticsProvider {
        require(shipmentLogisticsProviders[_shipmentId] == msg.sender, "Only the logistics provider assigned to the shipment can update its status");
        shipmentStatuses[_shipmentId] = _status;
        shipments[_shipmentId].updatedAt = block.timestamp;
        emit ShipmentStatusUpdate(_shipmentId, _status);
    }

    // Function to get a shipment by ID
    function getShipment(uint256 _shipmentId) public view returns (Shipment memory) {
        return shipments[_shipmentId];
    }

    // Function to get all shipments for a logistics provider
    function getShipmentsForLogisticsProvider(address _logisticsProvider) public view returns (Shipment[] memory) {
        Shipment[] memory shipmentsForProvider = new Shipment[](0);
        for (uint256 i = 0; i < shipments.length; i++) {
            if (shipments[i].logisticsProvider == _logisticsProvider) {
                shipmentsForProvider.push(shipments[i]);
            }
        }
        return shipmentsForProvider;
    }
}
