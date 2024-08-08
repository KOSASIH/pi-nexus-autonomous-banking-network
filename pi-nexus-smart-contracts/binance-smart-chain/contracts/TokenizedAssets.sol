// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC721/SafeERC721.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

contract TokenizedAssets {
    using Counters for Counters.Counter;
    using SafeERC721 for address;

    struct TokenizedAsset {
        address owner;
        uint256 tokenId;
        string uri;
    }

    mapping(uint256 => TokenizedAsset) public tokenizedAssets;
    Counters.Counter public tokenIdCounter;

    function createTokenizedAsset(string memory uri) public {
        require(uri != "", "Invalid URI");

        TokenizedAsset memory asset;
        asset.owner = msg.sender;
        asset.tokenId = tokenIdCounter.current();
        asset.uri = uri;

        tokenizedAssets[asset.tokenId] = asset;
        tokenIdCounter.increment();
    }

    function transferTokenizedAsset(uint256 tokenId, address to) public {
        require(tokenId > 0, "Invalid token ID");
        require(to != address(0), "Invalid recipient address");

        TokenizedAsset storage asset = tokenizedAssets[tokenId];
        require(asset.owner == msg.sender, "Not the owner of the tokenized asset");

        asset.owner = to;
    }

    function getOwner(uint256 tokenId) public view returns (address) {
        TokenizedAsset storage asset = tokenizedAssets[tokenId];
        return asset.owner;
    }

    function getURI(uint256 tokenId) public view returns (string memory) {
        TokenizedAsset storage asset = tokenizedAssets[tokenId];
        return asset.uri;
    }
}
