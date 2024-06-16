pragma solidity ^0.8.0;

import "./IPIBankNFTMarketplace.sol";

contract PIBankNFTMarketplace is IPIBankNFTMarketplace {
    mapping(uint256 => NFT) public nfts;
    uint256 public nftCount;

    struct NFT {
        string uri;
        address owner;
        uint256 price;
    }

    function mint(address _owner, string calldata _uri) public {
        NFT memory nft = NFT(_uri, _owner, 0);
        nfts[nftCount] = nft;
        nftCount++;
    }

    function buy(uint256 _nftId, uint256 _price) public {
        NFT storage nft = nfts[_nftId];
        require(nft.price == _price, "Invalid price");
        nft.owner = msg.sender;
    }

    function sell(uint256 _nftId, uint256 _price) public {
        NFT storage nft = nfts[_nftId];
        require(nft.owner == msg.sender, "Only the owner can sell");
        nft.price = _price;
    }

    function getNFT(uint256 _nftId) public view returns (string memory, address, uint256) {
        NFT storage nft = nfts[_nftId];
        return (nft.uri, nft.owner, nft.price);
    }
}
