// AstralPlaneMarketplace.sol
pragma solidity ^0.8.0;

contract AstralPlaneMarketplace {
    address private owner;
    mapping (address => mapping (address => uint256)) public listings;
    mapping (address => mapping (address => uint256)) public orders;

    constructor() public {
        owner = msg.sender;
    }

    function createListing(address asset, uint256 price) public {
        require(msg.sender == owner, "Only the owner can create listings");
        listings[msg.sender][asset] = price;
    }

    function buyAsset(address asset, uint256 amount) public {
        require(listings[msg.sender][asset] > 0, "Asset not listed for sale");
        require(amount <= listings[msg.sender][asset], "Insufficient amount listed");
        listings[msg.sender][asset] -= amount;
        // Transfer asset to buyer
        AstralPlaneAsset(asset).transfer(msg.sender, amount);
        // Update orders
        orders[msg.sender][asset] += amount;
    }

    function sellAsset(address asset, uint256 amount) public {
        require(AstralPlaneAsset(asset).balanceOf(msg.sender) >= amount, "Insufficient balance");
        // Create a new listing
        listings[msg.sender][asset] = amount;
        // Update orders
        orders[msg.sender][asset] += amount;
    }

    function cancelListing(address asset) public {
        require(listings[msg.sender][asset] > 0, "No listing to cancel");
        listings[msg.sender][asset] = 0;
        // Update orders
        orders[msg.sender][asset] = 0;
    }

    function getListing(address asset) public view returns (uint256) {
        return listings[msg.sender][asset];
    }

    function getOrder(address asset) public view returns (uint256) {
        return orders[msg.sender][asset];
    }
}
