pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";

contract PiOracle is Ownable {
    // The target value for the Pi-based stablecoin
    uint256 public constant TARGET_VALUE = 314159000000000000000; // $314.159 in wei

    // Mapping of Pi prices
    mapping (uint256 => uint256) public piPrices;

    // Mapping of Pi price updates
    mapping (uint256 => uint256) public piPriceUpdates;

    // Event emitted when the Pi price is updated
    event UpdatePiPrice(uint256 indexed timestamp, uint256 price);

    // Event emitted when the Pi price is updated by the owner
    event OwnerUpdatePiPrice(uint256 indexed timestamp, uint256 price);

    // Event emitted when the Pi price is updated by an oracle node
    event OracleUpdatePiPrice(uint256 indexed timestamp, uint256 price, address indexed oracleNode);

    // List of authorized oracle nodes
    mapping (address => bool) public oracleNodes;

    // Function to add an oracle node
    function addOracleNode(address _oracleNode) public onlyOwner {
        oracleNodes[_oracleNode] = true;
    }

    // Function to remove an oracle node
    function removeOracleNode(address _oracleNode) public onlyOwner {
        oracleNodes[_oracleNode] = false;
    }

    // Function to update the Pi price by the owner
    function updatePiPrice(uint256 _timestamp, uint256 _price) public onlyOwner {
        piPrices[_timestamp] = _price;
        piPriceUpdates[_timestamp] = block.timestamp;
        emit OwnerUpdatePiPrice(_timestamp, _price);
    }

    // Function to update the Pi price by an oracle node
    function updatePiPriceByOracle(uint256 _timestamp, uint256 _price) public {
        require(oracleNodes[msg.sender], "Only authorized oracle nodes can update the Pi price");
        piPrices[_timestamp] = _price;
        piPriceUpdates[_timestamp] = block.timestamp;
        emit OracleUpdatePiPrice(_timestamp, _price, msg.sender);
    }

    // Function to get the current Pi price
    function getPiPrice() public view returns (uint256) {
        return piPrices[block.timestamp];
    }

    // Function to get the historical Pi price at a specific timestamp
    function getPiPriceAt(uint256 _timestamp) public view returns (uint256) {
        return piPrices[_timestamp];
    }
}
