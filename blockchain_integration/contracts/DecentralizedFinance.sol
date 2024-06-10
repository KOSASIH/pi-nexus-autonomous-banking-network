pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract DecentralizedFinance {
    // Mapping of asset addresses to asset balances
    mapping (address => uint256) public assetBalances;

    // Event emitted when a newasset is added
    event NewAssetAdded(address assetAddress, uint256 initialBalance);

    // Function to add a new asset
    function addNewAsset(address _assetAddress, uint256 _initialBalance) public {
        // Add new asset
        assetBalances[_assetAddress] = _initialBalance;

        // Emit new asset added event
        emit NewAssetAdded(_assetAddress, _initialBalance);
    }

    // Function to transfer an asset
    function transferAsset(address _from, address _to, uint256 _amount) public {
        // Check if asset exists
        require(assetBalances[_from] != 0, "Asset does not exist");

        // Check if sender has enough balance
        require(assetBalances[_from] >= _amount, "Insufficient balance");

        // Transfer asset
        assetBalances[_from] = assetBalances[_from].sub(_amount);
        assetBalances[_to] = assetBalances[_to].add(_amount);
    }

    // Function to get an asset balance
    function getAssetBalance(address _assetAddress) public view returns (uint256) {
        return assetBalances[_assetAddress];
    }
}
