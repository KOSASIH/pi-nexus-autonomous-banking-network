// Escrow.sol

pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";

contract Escrow {
    using SafeERC20 for address;

    // Mapping of escrow accounts
    mapping (address => mapping (address => uint256)) public escrowBalances;

    // Mapping of escrow metadata
    mapping (address => mapping (address => EscrowMetadata)) public escrowMetadata;

    // Event emitted when an escrow account is created
    event EscrowCreated(address indexed sender, address indexed recipient, uint256 amount);

    // Event emitted when an escrow account is released
    event EscrowReleased(address indexed sender, address indexed recipient, uint256 amount);

    // Event emitted when an escrow account is disputed
    event EscrowDisputed(address indexed sender, address indexed recipient, uint256 amount);

    // Struct to store escrow metadata
    struct EscrowMetadata {
        uint256 createdAt;
        uint256 expiresAt;
        bool isDisputed;
    }

    // Modifier to prevent reentrancy attacks
    modifier nonReentrant() {
        require(!_isReentrant, "Reentrancy detected");
        _isReentrant = true;
        _;
        _isReentrant = false;
    }

    // Create a new escrow account
    function createEscrow(address recipient, uint256 amount) public nonReentrant {
        require(amount > 0, "Invalid amount");
        escrowBalances[msg.sender][recipient] = amount;
        escrowMetadata[msg.sender][recipient] = EscrowMetadata(block.timestamp, block.timestamp + 30 days, false);
        emit EscrowCreated(msg.sender, recipient, amount);
    }

    // Release an escrow account
    function releaseEscrow(address recipient) public nonReentrant {
        require(escrowBalances[msg.sender][recipient] > 0, "No escrow account found");
        require(!escrowMetadata[msg.sender][recipient].isDisputed, "Escrow account is disputed");
        recipient.safeTransfer(escrowBalances[msg.sender][recipient]);
        delete escrowBalances[msg.sender][recipient];
        delete escrowMetadata[msg.sender][recipient];
        emit EscrowReleased(msg.sender, recipient, escrowBalances[msg.sender][recipient]);
    }

    // Dispute an escrow account
    function disputeEscrow(address recipient) public nonReentrant {
        require(escrowBalances[msg.sender][recipient] > 0, "No escrow account found");
        escrowMetadata[msg.sender][recipient].isDisputed = true;
        emit EscrowDisputed(msg.sender, recipient, escrowBalances[msg.sender][recipient]);
    }

    // Resolve a disputed escrow account
    function resolveDispute(address recipient) public nonReentrant {
        require(escrowBalances[msg.sender][recipient] > 0, "No escrow account found");
        require(escrowMetadata[msg.sender][recipient].isDisputed, "Escrow account is not disputed");
        // Resolve the dispute using a custom dispute resolution mechanism (e.g. arbitration, voting, etc.)
        // ...
        delete escrowMetadata[msg.sender][recipient].isDisputed;
    }
}
