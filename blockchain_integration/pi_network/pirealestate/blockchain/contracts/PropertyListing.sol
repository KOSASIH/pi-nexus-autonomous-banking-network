pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC721/SafeERC721.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract PropertyListing {
    // Mapping of property listings to their owners
    mapping (address => mapping (uint256 => Property)) public propertyListings;

    // Mapping of property IDs to their metadata
    mapping (uint256 => PropertyMetadata) public propertyMetadata;

    // Event emitted when a new property listing is created
    event NewPropertyListing(address owner, uint256 propertyId, string metadata);

    // Event emitted when a property listing is updated
    event UpdatePropertyListing(address owner, uint256 propertyId, string metadata);

    // Event emitted when a property listing is deleted
    event DeletePropertyListing(address owner, uint256 propertyId);

    // Struct to represent a property listing
    struct Property {
        uint256 id;
        string metadata;
        address owner;
    }

    // Struct to represent property metadata
    struct PropertyMetadata {
        string name;
        string description;
        string location;
        uint256 price;
    }

    // Function to create a new property listing
    function createPropertyListing(string memory metadata) public {
        // Generate a unique property ID
        uint256 propertyId = uint256(keccak256(abi.encodePacked(msg.sender, metadata)));

        // Create a new property listing
        propertyListings[msg.sender][propertyId] = Property(propertyId, metadata, msg.sender);

        // Create a new property metadata
        propertyMetadata[propertyId] = PropertyMetadata("", "", "", 0);

        // Emit an event to notify the creation of a new property listing
        emit NewPropertyListing(msg.sender, propertyId, metadata);
    }

    // Function to update a property listing
    function updatePropertyListing(uint256 propertyId, string memory metadata) public {
        // Check if the property listing exists
        require(propertyListings[msg.sender][propertyId].id != 0, "Property listing does not exist");

        // Update the property listing
        propertyListings[msg.sender][propertyId].metadata = metadata;

        // Emit an event to notify the update of a property listing
        emit UpdatePropertyListing(msg.sender, propertyId, metadata);
    }

    // Function to delete a property listing
    function deletePropertyListing(uint256 propertyId) public {
        // Check if the property listing exists
        require(propertyListings[msg.sender][propertyId].id != 0, "Property listing does not exist");

        // Delete the property listing
        delete propertyListings[msg.sender][propertyId];

        // Emit an event to notify the deletion of a property listing
        emit DeletePropertyListing(msg.sender, propertyId);
    }

    // Function to get a property listing
    function getPropertyListing(uint256 propertyId) public view returns (Property memory) {
        // Check if the property listing exists
        require(propertyListings[msg.sender][propertyId].id != 0, "Property listing does not exist");

        // Return the property listing
        return propertyListings[msg.sender][propertyId];
    }

    // Function to get property metadata
    function getPropertyMetadata(uint256 propertyId) public view returns (PropertyMetadata memory) {
        // Check if the property metadata exists
        require(propertyMetadata[propertyId].name != "", "Property metadata does not exist");

        // Return the property metadata
        return propertyMetadata[propertyId];
    }
}
