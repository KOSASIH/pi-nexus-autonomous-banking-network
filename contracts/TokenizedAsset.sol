// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/Counters.sol";
import "@openzeppelin/contracts/token/common/ERC2981.sol";

contract TokenizedAsset is ERC721URIStorage, AccessControl, ERC2981 {
    using Counters for Counters.Counter;
    Counters.Counter private _tokenIdCounter;

    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");

    struct Asset {
        string name;
        string description;
        string uri; // URI for off-chain metadata
        address creator; // Address of the asset creator
        uint256 royalty; // Royalty percentage in basis points
    }

    mapping(uint256 => Asset) private _assets;

    event AssetMinted(uint256 indexed tokenId, address indexed owner, string name, string description, string uri, uint256 royalty);
    event AssetTransferred(uint256 indexed tokenId, address indexed from, address indexed to);
    event RoyaltyUpdated(uint256 indexed tokenId, uint256 newRoyalty);

    constructor() ERC721("TokenizedAsset", "TAS") {
        _setupRole(DEFAULT_ADMIN_ROLE, msg.sender);
    }

    // Function to mint a new tokenized asset
    function mintAsset(
        address to,
        string memory name,
        string memory description,
        string memory uri,
        uint256 royalty
    ) public onlyRole(MINTER_ROLE) {
        require(royalty <= 10000, "Royalty cannot exceed 100%"); // 100% in basis points

        uint256 tokenId = _tokenIdCounter.current();
        _mint(to, tokenId);
        _setTokenURI(tokenId, uri);
        
        _assets[tokenId] = Asset(name, description, uri, msg.sender, royalty);
        _setDefaultRoyalty(msg.sender, royalty);

        _tokenIdCounter.increment();

        emit AssetMinted(tokenId, to, name, description, uri, royalty);
    }

    // Function to batch mint assets
    function batchMintAssets(
        address[] memory recipients,
        string[] memory names,
        string[] memory descriptions,
        string[] memory uris,
        uint256[] memory royalties
    ) public onlyRole(MINTER_ROLE) {
        require(recipients.length == names.length && recipients.length == descriptions.length && recipients.length == uris.length && recipients.length == royalties.length, "Array lengths must match");

        for (uint256 i = 0; i < recipients.length; i++) {
            mintAsset(recipients[i], names[i], descriptions[i], uris[i], royalties[i]);
        }
    }

    // Function to get asset details
    function getAssetDetails(uint256 tokenId) public view returns (string memory name, string memory description, string memory uri, address creator, uint256 royalty) {
        require(_exists(tokenId), "Asset does not exist");
        Asset memory asset = _assets[tokenId];
        return (asset.name, asset.description, asset.uri, asset.creator, asset.royalty);
    }

    // Function to update the URI of an asset
    function updateAssetURI(uint256 tokenId, string memory newUri) public {
        require(ownerOf(tokenId) == msg.sender, "You do not own this asset");
        _setTokenURI(tokenId, newUri);
    }

    // Override the _transfer function to emit an event on transfer
    function _transfer(address from, address to, uint256 tokenId) internal override {
        super._transfer(from, to, tokenId);
        emit AssetTransferred(tokenId, from, to);
    }

    // Function to set royalties for an asset
    function setRoyalty(uint256 tokenId, uint256 newRoyalty) public {
        require(ownerOf(tokenId) == msg.sender, "You do not own this asset");
        require(newRoyalty <= 10000, "Royalty cannot exceed 100 %"); // 100% in basis points
        _assets[tokenId].royalty = newRoyalty;
        _setDefaultRoyalty(msg.sender, newRoyalty);
        emit RoyaltyUpdated(tokenId, newRoyalty);
    }

    // Function to transfer ownership of the asset
    function transferAsset(address to, uint256 tokenId) public {
        require(ownerOf(tokenId) == msg.sender, "You do not own this asset");
        _transfer(msg.sender, to, tokenId);
    }

    // Override the supportsInterface function to include ERC2981
    function supportsInterface(bytes4 interfaceId) public view virtual override(ERC721, AccessControl, ERC2981) returns (bool) {
        return super.supportsInterface(interfaceId);
    }
}
