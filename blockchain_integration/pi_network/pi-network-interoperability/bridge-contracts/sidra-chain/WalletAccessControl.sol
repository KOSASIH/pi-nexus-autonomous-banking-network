pragma solidity ^0.8.0;

contract WalletAccessControl {
    address public owner;
    mapping(address => bool) public allowedWallets;

    event WalletAdded(address indexed wallet);
    event WalletRemoved(address indexed wallet);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function");
        _;
    }

    constructor() public {
        owner = msg.sender;
    }

    function addWallet(address _wallet) public onlyOwner {
        allowedWallets[_wallet] = true;
        emit WalletAdded(_wallet);
    }

    function removeWallet(address _wallet) public onlyOwner {
        allowedWallets[_wallet] = false;
        emit WalletRemoved(_wallet);
    }

    function isWalletAllowed(address _wallet) public view returns (bool) {
        return allowedWallets[_wallet];
    }
}
