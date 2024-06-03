// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC721/ERC721.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Strings.sol";

contract NFT is ERC721 {
    using Strings for uint256;

    // Mapping of token IDs to their corresponding metadata
    mapping(uint256 => string) public tokenMetadata;

    // Event emitted when a new NFT is minted
    event Mint(address user, uint256 tokenId, string metadata);

    // Function to mint a new NFT
    function mint(address to, string memory metadata) public {
        // Generate a new token ID
        uint256 tokenId = totalSupply().add(1);

        // Mint the new NFT
        _mint(to, tokenId);

        // Set the metadata for the new NFT
        tokenMetadata[tokenId] = metadata;

        emit Mint(to, tokenId, metadata);
    }

    // Function to get the metadata for a token ID
    function tokenMetadata(uint256 tokenId) public view returns (string memory) {
        return tokenMetadata[tokenId];
    }
}
