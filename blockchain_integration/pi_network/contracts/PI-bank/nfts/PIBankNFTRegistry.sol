pragma solidity ^0.8.0;

import "./IPIBankNFTRegistry.sol";

contract PIBankNFTRegistry is IPIBankNFTRegistry {
    mapping(uint256 => NFT) public nfts;
    uint256 public nftCount;

    struct NFT {
        string uri;
        address owner;
    }

    function mint(address _owner, string calldata _uri) public {
        NFT memory nft = NFT(_uri, _owner);
        nfts[nftCount] = nft;
        nftCount++;
    }

    function burn(uint256 _nftId) public {
        NFT storage nft = nfts[_nftId];
        require(nft.owner == msg.sender, "Only the owner can burn");
        delete nfts[_nftId];
    }

    function transfer(address _from, address _to, uint256 _nftId) public {
        NFT storage nft = nfts[_nftId];
        require(nft.owner == _from, "Invalid owner");
        nft.owner = _to;
    }

    function ownerOf(uint256 _nftId) public view returns (address) {
        NFT storage nft = nfts[_nftId];
        return nft.owner;
    }

    function getNFT(uint256 _nftId) public view returns (string memory) {
        NFT storage nft = nfts[_nftId];
        return nft.uri;
    }
}
