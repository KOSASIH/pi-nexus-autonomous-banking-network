// TokenizedAssets.sol

pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC721/SafeERC721.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";

contract TokenizedAssets {
    using SafeERC721 for address;

    // Mapping of tokenized assets
    mapping (address => TokenizedAsset) public tokenizedAssets;

    // Event emitted when a new tokenized asset is created
    event TokenizedAssetCreated(address indexed creator, address indexed asset, uint256 tokenId);

    // Event emitted when a tokenized asset is transferred
    event TokenizedAssetTransferred(address indexed from, address indexed to, uint256 tokenId);

    // Struct to store tokenized asset metadata
    struct TokenizedAsset {
        address asset;
        uint256 tokenId;
        address owner;
    }

    // Modifier to prevent reentrancy attacks
    modifier nonReentrant() {
        require(!_isReentrant, "Reentrancy detected");
        _isReentrant = true;
        _;
        _isReentrant = false;
    }

    // Create a new tokenized asset
    function createTokenizedAsset(address asset, uint256 tokenId) public nonReentrant {
        require(asset != address(0), "Invalid asset");
        require(tokenId > 0, "Invalid token ID");
        TokenizedAsset storage token = tokenizedAssets[msg.sender];
        token.asset = asset;
        token.tokenId = tokenId;
        token.owner = msg.sender;
        emit TokenizedAssetCreated(msg.sender, asset, tokenId);
    }

    // Transfer a tokenized asset
    function transferTokenizedAsset(address to, uint256 tokenId) public nonReentrant {
        require(to != address(0), "Invalid recipient");
        require(tokenId > 0, "Invalid token ID");
        TokenizedAsset storage token = tokenizedAssets[msg.sender];
        require(token.tokenId == tokenId, "Token ID mismatch");
        token.owner = to;
        emit TokenizedAssetTransferred(msg.sender, to, tokenId);
    }

    // Get the owner of a tokenized asset
    function getOwner(uint256 tokenId) public view returns (address) {
        TokenizedAsset storage token = tokenizedAssets[tokenId];
        return token.owner;
    }

    // Get the asset associated with a tokenized asset
    function getAsset(uint256 tokenId) public view returns (address) {
        TokenizedAsset storage token = tokenizedAssets[tokenId];
        return token.asset;
    }

    // Get the token ID associated with a tokenized asset
    function getTokenId(address asset) public view returns (uint256) {
        TokenizedAsset storage token = tokenizedAssets[asset];
        return token.tokenId;
    }

    // Burn a tokenized asset
    function burnTokenizedAsset(uint256 tokenId) public nonReentrant {
        require(tokenId > 0, "Invalid token ID");
        TokenizedAsset storage token = tokenizedAssets[tokenId];
        require(token.owner == msg.sender, "Only the owner can burn the token");
        delete tokenizedAssets[tokenId];
        emit TokenizedAssetBurned(tokenId);
    }

    // Event emitted when a tokenized asset is burned
    event TokenizedAssetBurned(uint256 indexed tokenId);
}
