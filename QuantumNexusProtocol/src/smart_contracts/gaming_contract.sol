// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract GamingContract {
    struct GameAsset {
        string name;
        address owner;
    }

    mapping(uint256 => GameAsset) public assets;
    uint256 public assetcount;

    event AssetCreated(uint256 indexed assetId, string name, address indexed owner);
    event AssetTransferred(uint256 indexed assetId, address indexed from, address indexed to);

    function createAsset(string memory name) external {
        assetCount++;
        assets[assetCount] = GameAsset(name, msg.sender);
        emit AssetCreated(assetCount, name, msg.sender);
    }

    function transferAsset(uint256 assetId, address to) external {
        require(assets[assetId].owner == msg.sender, "You do not own this asset");
        require(to != address(0), "Invalid address");

        assets[assetId].owner = to;
        emit AssetTransferred(assetId, msg.sender, to);
    }

    function getAssetOwner(uint256 assetId) external view returns (address) {
        return assets[assetId].owner;
    }
}
