pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract AutoListingProtocol {
    // Mapping of Pi Coin listings on international financial systems
    mapping (address => mapping (address => bool)) public listings;

    // Event emitted when a new listing is created
    event NewListing(address indexed piCoin, address indexed financialSystem);

    // Event emitted when a listing is updated
    event ListingUpdated(address indexed piCoin, address indexed financialSystem);

    // Event emitted when a listing is removed
    event ListingRemoved(address indexed piCoin, address indexed financialSystem);

    // Function to create a new listing
    function createListing(address piCoin, address financialSystem) public {
        require(piCoin != address(0), "Pi Coin address cannot be zero");
        require(financialSystem != address(0), "Financial system address cannot be zero");
        listings[piCoin][financialSystem] = true;
        emit NewListing(piCoin, financialSystem);
    }

    // Function to update a listing
    function updateListing(address piCoin, address financialSystem) public {
        require(piCoin != address(0), "Pi Coin address cannot be zero");
        require(financialSystem != address(0), "Financial system address cannot be zero");
        listings[piCoin][financialSystem] = true;
        emit ListingUpdated(piCoin, financialSystem);
    }

    // Function to remove a listing
    function removeListing(address piCoin, address financialSystem) public {
        require(piCoin != address(0), "Pi Coin address cannot be zero");
        require(financialSystem != address(0), "Financial system address cannot be zero");
        listings[piCoin][financialSystem] = false;
        emit ListingRemoved(piCoin, financialSystem);
    }

    // Function to get the listing status of a Pi Coin on a financial system
    function getListingStatus(address piCoin, address financialSystem) public view returns (bool) {
        return listings[piCoin][financialSystem];
    }
}
