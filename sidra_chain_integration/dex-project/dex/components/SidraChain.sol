pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Roles.sol";

contract SidraChain {
    // Mapping of user addresses to their respective balances
    mapping (address => uint256) public balances;

    // Mapping of asset IDs to their respective details
    mapping (uint256 => Asset) public assets;

    // Event emitted when a new asset is created
    event NewAsset(uint256 assetId, string assetName, uint256 assetValue);

    // Event emitted when a transaction is processed
    event TransactionProcessed(address from, address to, uint256 amount, uint256 assetId);

    // Role-based access control for administrators
    Roles.Role private administrators;

    // Constructor function
    constructor() public {
        administrators.add(msg.sender);
    }

    // Function to create a new asset
    function createAsset(string memory _assetName, uint256 _assetValue) public {
        require(administrators.has(msg.sender), "Only administrators can create assets");
        uint256 assetId = assets.length++;
        assets[assetId] = Asset(_assetName, _assetValue);
        emit NewAsset(assetId, _assetName, _assetValue);
    }

    // Function to process a transaction
    function processTransaction(address _from, address _to, uint256 _amount, uint256 _assetId) public {
        require(balances[_from] >= _amount, "Insufficient balance");
        balances[_from] -= _amount;
        balances[_to] += _amount;
        emit TransactionProcessed(_from, _to, _amount, _assetId);
    }

    // Function to get the balance of a user
    function getBalance(address _user) public view returns (uint256) {
        return balances[_user];
    }

    // Function to get the details of an asset
    function getAsset(uint256 _assetId) public view returns (string memory, uint256) {
        return (assets[_assetId].name, assets[_assetId].value);
    }

    // Struct to represent an asset
    struct Asset {
        string name;
        uint256 value;
    }
}
