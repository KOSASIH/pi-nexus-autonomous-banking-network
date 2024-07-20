// genesis-smart-contracts/WalletAccessControl.sol
pragma solidity ^0.8.0;

contract WalletAccessControl {
    mapping (address => bool) public walletAccess;

    function enableWallet(address wallet) public onlyOwner {
        walletAccess[wallet] = true;
    }

    function disableWallet(address wallet) public onlyOwner {
        walletAccess[wallet] = false;
    }

    function batchEnableWallets(address[] memory wallets) public onlyOwner {
        for (uint256 i = 0; i < wallets.length; i++) {
            walletAccess[wallets[i]] = true;
        }
    }

    function batchDisableWallets(address[] memory wallets) public onlyOwner {
        for (uint256 i = 0; i < wallets.length; i++) {
            walletAccess[wallets[i]] = false;
        }
    }
}
