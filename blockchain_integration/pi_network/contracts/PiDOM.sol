pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/cryptography/ECDSA.sol";

contract PiDOM {
    // Mapping of digital asset IDs to their respective assets
    mapping (bytes32 => DigitalAsset) public digitalAssets;

    // Mapping of user addresses to their respective digital asset listings
    mapping (address => mapping (bytes32 => DigitalAssetListing)) public digitalAssetListings;

    // Mapping of user addresses to their respective transaction histories
    mapping (address => Transaction[]) public transactionHistories;

    // Event emitted when a new digital asset is registered
    event DigitalAssetRegistered(bytes32 indexed digitalAssetId, DigitalAsset digitalAsset);

    // Event emitted when a digital asset is listed for sale
    event DigitalAssetListed(bytes32 indexed digitalAssetId, DigitalAssetListing digitalAssetListing);

    // Event emitted when a digital asset is traded
    event DigitalAssetTraded(bytes32 indexed digitalAssetId, address buyer, address seller, uint256 amount);

    // Event emitted when a user's reputation is updated
    event ReputationUpdated(address indexed user, uint256 reputation);

    struct DigitalAsset {
        bytes32 id;
        string name;
        string description;
        uint256 price;
        uint256 quantity;
        address owner;
    }

    struct DigitalAssetListing {
        bytes32 digitalAssetId;
        uint256 price;
        uint256 quantity;
        address seller;
    }

    struct Transaction {
        bytes32 digitalAssetId;
        address buyer;
        address seller;
        uint256 amount;
        uint256 timestamp;
    }

    /**
     * @dev Registers a new digital asset on the Pi Network
     * @param _digitalAsset The digital asset to register
     */
    function registerDigitalAsset(DigitalAsset _digitalAsset) public {
        require(digitalAssets[msg.sender][_digitalAsset.id] == 0, "Digital asset already exists");
        digitalAssets[msg.sender][_digitalAsset.id] = _digitalAsset;
        emit DigitalAssetRegistered(_digitalAsset.id, _digitalAsset);
    }

    /**
     * @dev Lists a digital asset for sale on the open market
     * @param _digitalAssetId The ID of the digital asset to list
     * @param _price The price of the digital asset
     * @param _quantity The quantity of the digital asset
     */
    function listDigitalAsset(bytes32 _digitalAssetId, uint256 _price, uint256 _quantity) public {
        require(digitalAssets[msg.sender][_digitalAssetId] != 0, "Digital asset does not exist");
        digitalAssetListings[msg.sender][_digitalAssetId] = DigitalAssetListing(_digitalAssetId, _price, _quantity, msg.sender);
        emit DigitalAssetListed(_digitalAssetId, digitalAssetListings[msg.sender][_digitalAssetId]);
    }

        /**
     * @dev Trades a digital asset between two users
     * @param _digitalAssetId The ID of the digital asset to trade
     * @param _buyer The buyer's address
     * @param _seller The seller's address
     * @param _amount The amount of the digital asset to trade
     */
    function tradeDigitalAsset(bytes32 _digitalAssetId, address _buyer, address _seller, uint256 _amount) public {
        require(digitalAssets[_seller][_digitalAssetId] != 0, "Digital asset does not exist");
        require(digitalAssetListings[_seller][_digitalAssetId].quantity >= _amount, "Insufficient quantity");
        digitalAssets[_buyer][_digitalAssetId] = digitalAssets[_seller][_digitalAssetId];
        digitalAssets[_seller][_digitalAssetId].quantity -= _amount;
        digitalAssetListings[_seller][_digitalAssetId].quantity -= _amount;
        transactionHistories[_buyer].push(Transaction(_digitalAssetId, _buyer, _seller, _amount, block.timestamp));
        transactionHistories[_seller].push(Transaction(_digitalAssetId, _buyer, _seller, _amount, block.timestamp));
        emit DigitalAssetTraded(_digitalAssetId, _buyer, _seller, _amount);
    }

    /**
     * @dev Updates a user's reputation
     * @param _user The user's address
     * @param _reputation The new reputation score
     */
    function updateReputation(address _user, uint256 _reputation) public {
        require(_user != address(0), "Invalid user address");
        reputation[_user] = _reputation;
        emit ReputationUpdated(_user, _reputation);
    }

    /**
     * @dev Gets a digital asset by ID
     * @param _digitalAssetId The ID of the digital asset to get
     * @return The digital asset with the specified ID
     */
    function getDigitalAsset(bytes32 _digitalAssetId) public view returns (DigitalAsset memory) {
        return digitalAssets[msg.sender][_digitalAssetId];
    }

    /**
     * @dev Gets a digital asset listing by ID
     * @param _digitalAssetId The ID of the digital asset listing to get
     * @return The digital asset listing with the specified ID
     */
    function getDigitalAssetListing(bytes32 _digitalAssetId) public view returns (DigitalAssetListing memory) {
        return digitalAssetListings[msg.sender][_digitalAssetId];
    }

    /**
     * @dev Gets a user's transaction history
     * @param _user The user's address
     * @return The user's transaction history
     */
    function getTransactionHistory(address _user) public view returns (Transaction[] memory) {
        return transactionHistories[_user];
    }

    /**
     * @dev Gets a user's reputation
     * @param _user The user's address
     * @return The user's reputation score
     */
    function getReputation(address _user) public view returns (uint256) {
        return reputation[_user];
    }
}
