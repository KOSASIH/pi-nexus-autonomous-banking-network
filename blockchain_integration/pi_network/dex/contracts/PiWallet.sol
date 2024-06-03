pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";
import "https://github.com/smartcontractkit/chainlink/blob/master/evm-contracts/src/v0.6/ChainlinkClient.sol";

contract PiWallet {
    using SafeMath for uint256;

    struct Wallet {
        address owner;
        uint256 balance;
        uint256 nonce;
    }

    mapping (address => Wallet) public wallets;

    event WalletCreated(address indexed owner, uint256 balance);
    event WalletUpdated(address indexed owner, uint256 balance);

    function createWallet() public {
        Wallet memory wallet = Wallet(msg.sender, 0, 0);
        wallets[msg.sender] = wallet;
        emit WalletCreated(msg.sender, 0);
    }

    function updateWalletBalance(uint256 amount) public {
        Wallet storage wallet = wallets[msg.sender];
        wallet.balance = wallet.balance.add(amount);
        emit WalletUpdated(msg.sender, wallet.balance);
    }

    function getWalletBalance(address owner) public view returns (uint256) {
        return wallets[owner].balance;
    }

    function getWalletNonce(address owner) public view returns (uint256) {
        return wallets[owner].nonce;
    }
}
