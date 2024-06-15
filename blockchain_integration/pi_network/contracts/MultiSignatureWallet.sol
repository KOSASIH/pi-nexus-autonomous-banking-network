pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/security/ReentrancyGuard.sol";

contract MultiSignatureWallet is ReentrancyGuard {
    using SafeMath for uint256;

    // Mapping of wallet owners
    mapping (address => bool) public owners;

    // Mapping of wallet balances
    mapping (address => uint256) public balances;

    // Event emitted when a new owner is added
    event NewOwner(address indexed owner);

    // Event emitted when a wallet balance is updated
    event BalanceUpdated(address indexed owner, uint256 indexed balance);

    // Event emitted when a transaction is processed
    event TransactionProcessed(address indexed from, address indexed to, uint256 amount);

    // Function to add a new owner
    function addOwner(address owner) public {
        require(!owners[owner], "Owner already exists");
        owners[owner] = true;
        emit NewOwner(owner);
    }

    // Function to remove an owner
    function removeOwner(address owner) public {
        require(owners[owner], "Owner does not exist");
        delete owners[owner];
    }

    // Function to update a wallet balance
    function updateBalance(address owner, uint256 newBalance) public {
        require(owners[msg.sender], "Unauthorized access");
        balances[owner] = newBalance;
        emit BalanceUpdated(owner, newBalance);
    }

    // Function to process a transaction
    function processTransaction(address from, address to, uint256 amount) public {
        require(owners[msg.sender], "Unauthorized access");
        require(balances[from] >= amount, "Insufficient balance");
        balances[from] = balances[from].sub(amount);
        balances[to] = balances[to].add(amount);
        emit TransactionProcessed(from, to, amount);
    }
}
