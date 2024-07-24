pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC721/SafeERC721.sol";

contract AstralPlaneNFT is SafeERC721 {
    struct AstralPlaneNFT {
        uint256 id;
        string name;
        string description;
        uint256 price;
        address owner;
    }

    mapping (uint256 => AstralPlaneNFT) public nfts;
    uint256 public nextNftId;

    event NFTCreated(uint256 id, string name, string description, uint256 price);
    event NFTTransferred(uint256 id, address from, address to);

    function createNFT(string memory name, string memory description, uint256 price) public {
        uint256 id = nextNftId++;
        nfts[id] = AstralPlaneNFT(id, name, description, price, msg.sender);
        emit NFTCreated(id, name, description, price);
    }

    function transferNFT(uint256 id, address to) public {
        require(nfts[id].owner == msg.sender, "Only the owner can transfer this NFT");
        nfts[id].owner = to;
        emit NFTTransferred(id, msg.sender, to);
    }

    function getNFT(uint256 id) public view returns (AstralPlaneNFT memory) {
        return nfts[id];
    }
}
