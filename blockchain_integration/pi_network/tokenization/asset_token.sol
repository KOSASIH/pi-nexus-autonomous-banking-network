// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract AssetToken is ERC20, Ownable {
    struct Asset {
        string name;
        string description;
        uint256 value; // in wei
        bool exists;
    }

    mapping(uint256 => Asset) public assets; // assetId => Asset
    mapping(uint256 => uint256) public assetTokens; // assetId => total tokens minted
    uint256 public nextAssetId;

    event AssetCreated(uint256 indexed assetId, string name, string description, uint256 value);
    event TokensMinted(uint256 indexed assetId, address indexed to, uint256 amount);
    event TokensBurned(uint256 indexed assetId, address indexed from, uint256 amount);

    constructor() ERC20("AssetToken", "ATKN") {}

    // Create a new asset
    function createAsset(string memory _name, string memory _description, uint256 _value) public onlyOwner {
        assets[nextAssetId] = Asset(_name, _description, _value, true);
        emit AssetCreated(nextAssetId, _name, _description, _value);
        nextAssetId++;
    }

    // Mint tokens for an asset
    function mintTokens(uint256 _assetId, address _to, uint256 _amount) public onlyOwner {
        require(assets[_assetId].exists, "Asset does not exist.");
        _mint(_to, _amount);
        assetTokens[_assetId] += _amount;
        emit TokensMinted(_assetId, _to, _amount);
    }

    // Burn tokens for an asset
    function burnTokens(uint256 _assetId, uint256 _amount) public {
        require(assets[_assetId].exists, "Asset does not exist.");
        _burn(msg.sender, _amount);
        assetTokens[_assetId] -= _amount;
        emit TokensBurned(_assetId, msg.sender, _amount);
    }

    // Get asset details
    function getAssetDetails(uint256 _assetId) public view returns (string memory, string memory, uint256, bool) {
        Asset storage asset = assets[_assetId];
        return (asset.name, asset.description, asset.value, asset.exists);
    }
}
