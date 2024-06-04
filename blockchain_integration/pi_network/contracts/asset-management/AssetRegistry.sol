pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC721/ERC721.sol";

contract AssetRegistry {
    mapping(string => address) public assets;

    function registerAsset(string memory assetName, address assetAddress) public {
        require(assets[assetName] == address(0), "Asset already registered");
        assets[assetName] = assetAddress;
    }

    function getAssetAddress(string memory assetName) public view returns (address) {
        return assets[assetName];
    }
}
