pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC721/SafeERC721.sol";

contract AstralPlaneAsset is ERC721 {
    address private owner;
    mapping (address => uint256) public assetBalances;
    mapping (uint256 => Asset) public assets;

    struct Asset {
        uint256 id;
        string name;
        string description;
        string image;
        uint256 price;
    }

    event AssetCreated(uint256 id, string name, string description, string image, uint256 price);
    event AssetBought(uint256 id, address buyer, uint256 price);

    constructor() public {
        owner = msg.sender;
    }

    function createAsset(string memory _name, string memory _description, string memory _image, uint256 _price) public {
        require(msg.sender == owner, "Only the owner can create assets");
        uint256 id = assets.length++;
        assets[id] = Asset(id, _name, _description, _image, _price);
        emit AssetCreated(id, _name, _description, _image, _price);
    }

    function buyAsset(uint256 _id) public payable {
        require(msg.sender!= owner, "The owner cannot buy assets");
        require(msg.value >= assets[_id].price, "Insufficient funds");
        assetBalances[msg.sender]++;
        emit AssetBought(_id, msg.sender, assets[_id].price);
    }

    function getAssets() public view returns (Asset[] memory) {
        Asset[] memory _assets = new Asset[](assets.length);
        for (uint256 i = 0; i < assets.length; i++) {
            _assets[i] = assets[i];
        }
        return _assets;
    }
}
