pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC721/SafeERC721.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Counters.sol";

contract DynamicNFTMarketplace {
    using SafeERC721 for ERC721;
    using Counters for Counters.Counter;

    // Mapping of NFT IDs to attributes
    mapping (uint256 => Attribute) public nftAttributes;

    // Mapping of user addresses to NFT balances
    mapping (address => mapping (uint256 => uint256)) public nftBalances;

    // Event emitted when an NFT is created
    event NFTCreated(uint256 nftId, address creator, Attribute attributes);

    // Event emitted when an NFT is bought
    event NFTBought(uint256 nftId, address buyer, uint256 price);

    // Event emitted when an NFT is sold
    event NFTSold(uint256 nftId, address seller, uint256 price);

    // Struct to represent NFT attributes
    struct Attribute {
        string name;
        string description;
        uint256 rarity;
        uint256 level;
    }

    // Function to create an NFT
    function createNFT(Attribute memory attributes) public {
        uint256 nftId = Counters.Counter(nftIdCounter).current();
        nftAttributes[nftId] = attributes;
        nftBalances[msg.sender][nftId] = 1;
        emit NFTCreated(nftId, msg.sender, attributes);
    }

    // Function to buy an NFT
    function buyNFT(uint256 nftId, uint256 price) public {
        require(nftBalances[ownerOf(nftId)][nftId] == 1, "NFT not available");
        require(ERC721(ownerOf(nftId)).transferFrom(ownerOf(nftId), msg.sender, nftId), "Transfer failed");
        nftBalances[msg.sender][nftId] = 1;
        nftBalances[ownerOf(nftId)][nftId] = 0;
        emit NFTBought(nftId, msg.sender, price);
    }

    // Function to sell an NFT
    function sellNFT(uint256 nftId, uint256 price) public {
        require(nftBalances[msg.sender][nftId] == 1, "NFT not owned");
        nftBalances[msg.sender][nftId] = 0;
        nftBalances[ownerOf(nftId)][nftId] = 1;
        emit NFTSold(nftId, msg.sender, price);
    }
}
