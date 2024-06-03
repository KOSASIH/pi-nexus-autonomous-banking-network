pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";

contract PiBridge {
    using SafeMath for uint256;

    // Mapping of assets
    mapping (address => Asset) public assets;

    // Event emitted when a new asset is added
    event AssetAdded(address indexed asset, uint256 amount);

    // Event emitted when an asset is transferred
    event AssetTransferred(address indexed from, address indexed to, uint256 amount);

    // Struct to represent an asset
    struct Asset {
        address asset;
        uint256 amount;
        uint256 timestamp;
    }

    // Function to add a new asset
    function addAsset(address asset, uint256 amount) public {
        // Create a new asset
        Asset memory asset = Asset(asset, amount, block.timestamp);
        assets[asset] = asset;

        // Emit the AssetAdded event
        emit AssetAdded(asset, amount);
    }

    // Function to transfer an asset
    function transferAsset(address from, address to, uint256 amount) public {
        // Get the assets
        Asset storage f = assets[from];
        Asset storage t = assets[to];

        // Check if the assets exist
        require(f.amount > 0, "From asset does not exist");
        require(t.amount > 0, "To asset does not exist");

        // Transfer the assets
        f.amount = f.amount.sub(amount);
        t.amount = t.amount.add(amount);

        // Emit the AssetTransferred event
        emit AssetTransferred(from, to, amount);
    }
}
