// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

contract NFTMarketplace is ERC721 {
    using Counters for Counters.Counter;
    Counters.Counter private _tokenIdCounter;

    struct NFT {
        uint256 id;
        address owner;
        uint256 price;
        bool isListed;
    }

    mapping(uint256 => NFT) public nfts;

    constructor() ERC721("MyNFT", "MNFT") {}

    function mintNFT(uint256 price) external {
        _tokenIdCounter.increment();
        uint256 newItemId = _tokenIdCounter.current();
        _mint(msg.sender, newItemId);
        nfts[newItemId] = NFT(newItemId, msg.sender, price, false);
    }

    function listNFT(uint256 tokenId, uint256 price) external {
        require(ownerOf(tokenId) == msg.sender, "Not the owner");
        nfts[tokenId].price = price;
        nfts[tokenId].isListed = true;
    }

    function buyNFT(uint256 tokenId) external payable {
        require(nfts[tokenId].isListed, "NFT not listed");
        require(msg.value >= nfts[tokenId].price, "Insufficient funds");

        address seller = nfts[tokenId].owner;
        payable(seller).transfer(msg.value);
        _transfer(seller, msg.sender, tokenId);
        nfts[tokenId].owner = msg.sender;
        nfts[tokenId].isListed = false;
    }
}
