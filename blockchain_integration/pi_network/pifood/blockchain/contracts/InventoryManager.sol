pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC721/SafeERC721.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Address.sol";

contract InventoryManager {
    using SafeERC721 for address;
    using SafeMath for uint256;
    using Address for address;

    // Mapping of inventory items to their owners
    mapping (address => mapping (uint256 => InventoryItem)) public inventory;

    // Mapping of inventory item IDs to their metadata
    mapping (uint256 => InventoryItemMetadata) public inventoryItemMetadata;

    // Event emitted when an inventory item is created
    event InventoryItemCreated(uint256 indexed itemId, address indexed owner);

    // Event emitted when an inventory item is updated
    event InventoryItemUpdated(uint256 indexed itemId, address indexed owner);

    // Event emitted when an inventory item is deleted
    event InventoryItemDeleted(uint256 indexed itemId, address indexed owner);

    // Event emitted when an inventory item is transferred
    event InventoryItemTransferred(uint256 indexed itemId, address indexed from, address indexed to);

    // Struct to represent an inventory item
    struct InventoryItem {
        uint256 id;
        string name;
        string description;
        uint256 quantity;
        address owner;
    }

    // Struct to represent inventory item metadata
    struct InventoryItemMetadata {
        string imageURI;
        string attributes;
    }

    // Function to create a new inventory item
    function createInventoryItem(string memory _name, string memory _description, uint256 _quantity) public {
        // Generate a unique ID for the inventory item
        uint256 itemId = uint256(keccak256(abi.encodePacked(_name, _description, _quantity)));

        // Create a new inventory item
        InventoryItem memory inventoryItem = InventoryItem(itemId, _name, _description, _quantity, msg.sender);

        // Add the inventory item to the mapping
        inventory[msg.sender][itemId] = inventoryItem;

        // Add the inventory item metadata to the mapping
        inventoryItemMetadata[itemId] = InventoryItemMetadata("", "");

        // Emit an event to notify that a new inventory item has been created
        emit InventoryItemCreated(itemId, msg.sender);
    }

    // Function to update an inventory item
    function updateInventoryItem(uint256 _itemId, string memory _name, string memory _description, uint256 _quantity) public {
        // Check if the inventory item exists
        require(inventory[msg.sender][_itemId].id != 0, "Inventory item does not exist");

        // Update the inventory item
        inventory[msg.sender][_itemId].name = _name;
        inventory[msg.sender][_itemId].description = _description;
        inventory[msg.sender][_itemId].quantity = _quantity;

        // Emit an event to notify that an inventory item has been updated
        emit InventoryItemUpdated(_itemId, msg.sender);
    }

    // Function to delete an inventory item
    function deleteInventoryItem(uint256 _itemId) public {
        // Check if the inventory item exists
        require(inventory[msg.sender][_itemId].id != 0, "Inventory item does not exist");

        // Delete the inventory item
        delete inventory[msg.sender][_itemId];

        // Emit an event to notify that an inventory item has been deleted
        emit InventoryItemDeleted(_itemId, msg.sender);
    }

    // Function to transfer an inventory item
    function transferInventoryItem(uint256 _itemId, address _to) public {
        // Check if the inventory item exists
        require(inventory[msg.sender][_itemId].id != 0, "Inventory item does not exist");

        // Transfer the inventory item to the new owner
        inventory[_to][_itemId] = inventory[msg.sender][_itemId];
        inventory[msg.sender][_itemId].owner = _to;

        // Emit an event to notify that an inventory item has been transferred
        emit InventoryItemTransferred(_itemId, msg.sender, _to);
    }

    // Function to get an inventory item
    function getInventoryItem(uint256 _itemId) public view returns (InventoryItem memory) {
        // Check if the inventory item exists
        require(inventory[msg.sender][_itemId].id != 0, "Inventory item does not exist");

        // Return the inventory item
        return inventory[msg.sender][_itemId];
    }

    // Function to get inventory item metadata
    function getInventoryItemMetadata(uint256 _itemId) public view returns (InventoryItemMetadata memory) {
        // Check if the inventory item exists
        require(inventory[msg.sender][_itemId].id != 0, "Inventory item does not exist");

        // Return the inventory item metadata
        return inventoryItemMetadata[_itemId];
       // Function to update inventory item metadata
    function updateInventoryItemMetadata(uint256 _itemId, string memory _imageURI, string memory _attributes) public {
        // Check if the inventory item exists
        require(inventory[msg.sender][_itemId].id != 0, "Inventory item does not exist");

        // Update the inventory item metadata
        inventoryItemMetadata[_itemId].imageURI = _imageURI;
        inventoryItemMetadata[_itemId].attributes = _attributes;
    }

    // Function to get the balance of an inventory item
    function getInventoryItemBalance(uint256 _itemId) public view returns (uint256) {
        // Check if the inventory item exists
        require(inventory[msg.sender][_itemId].id != 0, "Inventory item does not exist");

        // Return the balance of the inventory item
        return inventory[msg.sender][_itemId].quantity;
    }

    // Function to increment the balance of an inventory item
    function incrementInventoryItemBalance(uint256 _itemId, uint256 _amount) public {
        // Check if the inventory item exists
        require(inventory[msg.sender][_itemId].id != 0, "Inventory item does not exist");

        // Increment the balance of the inventory item
        inventory[msg.sender][_itemId].quantity = inventory[msg.sender][_itemId].quantity.add(_amount);
    }

    // Function to decrement the balance of an inventory item
    function decrementInventoryItemBalance(uint256 _itemId, uint256 _amount) public {
        // Check if the inventory item exists
        require(inventory[msg.sender][_itemId].id != 0, "Inventory item does not exist");

        // Decrement the balance of the inventory item
        inventory[msg.sender][_itemId].quantity = inventory[msg.sender][_itemId].quantity.sub(_amount);
    }

    // Function to get the total balance of all inventory items
    function getTotalInventoryBalance() public view returns (uint256) {
        uint256 totalBalance = 0;

        // Iterate over all inventory items
        for (uint256 i = 0; i < inventory[msg.sender].length; i++) {
            totalBalance = totalBalance.add(inventory[msg.sender][i].quantity);
        }

        // Return the total balance
        return totalBalance;
    }

    // Function to get the total value of all inventory items
    function getTotalInventoryValue() public view returns (uint256) {
        uint256 totalValue = 0;

        // Iterate over all inventory items
        for (uint256 i = 0; i < inventory[msg.sender].length; i++) {
            totalValue = totalValue.add(inventory[msg.sender][i].quantity.mul(inventoryItemMetadata[i].attributes));
        }

        // Return the total value
        return totalValue;
    }
}
