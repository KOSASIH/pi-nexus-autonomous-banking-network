pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC721/SafeERC721.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract PropertyTransfer {
    // Mapping of property owners to their properties
    mapping (address => mapping (uint256 => Property)) public propertyOwners;

    // Mapping of property IDs to their metadata
    mapping (uint256 => PropertyMetadata) public propertyMetadata;

    // Event emitted when a property is transferred
    event TransferProperty(address from, address to, uint256 propertyId);

    // Event emitted when a property is listed for sale
    event ListPropertyForSale(address owner, uint256 propertyId, uint256 price);

    // Event emitted when a property is sold
    event SellProperty(address buyer, uint256 propertyId, uint256 price);

    // Struct to represent a property
    struct Property {
        uint256 id;
        address owner;
    }

    // Struct to represent property metadata
    struct PropertyMetadata {
        string name;
        string description;
        string location;
        uint256 price;
    }

    // Function to transfer a property
    function transferProperty(uint256 propertyId, address to) public {
        // Check if the property exists
        require(propertyOwners[msg.sender][propertyId].id != 0, "Property does not exist");

        // Check if the sender is the owner of the property
        require(propertyOwners[msg.sender][propertyId].owner == msg.sender, "Sender is not the owner of the property");

        // Transfer the property
        propertyOwners[to][propertyId] = Property(propertyId, to);

        // Delete the property from the sender's ownership
        delete propertyOwners[msg.sender][propertyId];

        // Emit an event to notify the transfer of a property
        emit TransferProperty(msg.sender, to, propertyId);
    }

    // Function to list a property for sale
    function listPropertyForSale(uint256 propertyId, uint256 price) public {
        // Check if the property exists
        require(propertyOwners[msg.sender][propertyId].id != 0, "Property does not exist");

        // Check if the sender is the owner of the property
        require(propertyOwners[msg.sender][propertyId].owner == msg.sender, "Sender is not the owner of the property");

        // Update the property metadata with the sale price
        propertyMetadata[propertyId].price = price;

        // Emit an event to notify the listing of a property for sale
        emit ListPropertyForSale(msg.sender, propertyId, price);
    }

    // Function to sell a property
    function sellProperty(uint256 propertyId) public payable {
        // Check if the property exists
        require(propertyOwners[msg.sender][propertyId].id != 0, "Property does not exist");

        // Check if the sender is the owner of the property
        require(propertyOwners[msg.sender][propertyId].owner == msg.sender, "Sender is not the owner of the property");

        // Check if the sale price is valid
        require(msg.value >= propertyMetadata[propertyId].price, "Sale price is not valid");

        // Transfer the property to the buyer
        propertyOwners[msg.sender][propertyId].owner = msg.sender;

        // Update the property metadata with the new owner
        propertyMetadata[propertyId].owner = msg.sender;

        // Emit an event to notify the sale of a property
        emit SellProperty(msg.sender, propertyId, msg.value);
    }

    // Function to get a property owner
    function getPropertyOwner(uint256 propertyId) public view returns (address) {
        // Check if the property exists
        require(propertyOwners[msg.sender][propertyId].id != 0, "Property does not exist");

        // Return the property owner
        return propertyOwners[msg.sender][propertyId].owner;
    }

    // Function to get a property
    function getProperty(uint256 propertyId) public view returns (Property memory) {
        // Check if the property exists
        require(propertyOwners[msg.sender][propertyId].id != 0, "Property does not exist");

        // Return the property
        return propertyOwners[msg.sender][propertyId];
    }

    // Function to get property metadata
    function getPropertyMetadata(uint256 propertyId) public view returns (PropertyMetadata memory) {
        // Check if the property metadata exists
        require(propertyMetadata[propertyId].name != "", "Property metadata does not exist");

        // Return the property metadata
        return propertyMetadata[propertyId];
    }
}
