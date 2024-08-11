pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "./NexusToken.sol";

contract NexusContract is Ownable {
    using SafeMath for uint256;
    using SafeERC20 for ERC20;

    // Mapping of user addresses to their Nexus token balances
    mapping (address => uint256) public nexusBalances;

    // Mapping of user addresses to their Nexus token allowances
    mapping (address => mapping (address => uint256)) public nexusAllowances;

    // Event emitted when a user deposits Nexus tokens
    event Deposit(address indexed user, uint256 amount);

    // Event emitted when a user withdraws Nexus tokens
    event Withdrawal(address indexed user, uint256 amount);

    // Event emitted when a user transfers Nexus tokens
    event Transfer(address indexed from, address indexed to, uint256 amount);

    // Nexus token contract
    NexusToken public nexusToken;

    // Constructor
    constructor() public {
        nexusToken = NexusToken(NEXUS_TOKEN_ADDRESS);
    }

    // Deposit Nexus tokens
    function deposit(uint256 amount) public {
        require(amount > 0, "Deposit amount must be greater than 0");
        nexusToken.safeTransferFrom(msg.sender, address(this), amount);
        nexusBalances[msg.sender] = nexusBalances[msg.sender].add(amount);
        emit Deposit(msg.sender, amount);
    }

    // Withdraw Nexus tokens
    function withdraw(uint256 amount) public {
        require(amount > 0, "Withdrawal amount must be greater than 0");
        require(nexusBalances[msg.sender] >= amount, "Insufficient balance");
        nexusToken.safeTransfer(msg.sender, amount);
        nexusBalances[msg.sender] = nexusBalances[msg.sender].sub(amount);
        emit Withdrawal(msg.sender, amount);
    }

    // Transfer Nexus tokens
    function transfer(address to, uint256 amount) public {
        require(to != address(0), "Cannot transfer to zero address");
        require(amount > 0, "Transfer amount must be greater than 0");
        require(nexusBalances[msg.sender] >= amount, "Insufficient balance");
        nexusBalances[msg.sender] = nexusBalances[msg.sender].sub(amount);
        nexusBalances[to] = nexusBalances[to].add(amount);
        emit Transfer(msg.sender, to, amount);
    }

    // Approve Nexus token allowance
    function approve(address spender, uint256 amount) public {
        require(spender != address(0), "Cannot approve zero address");
        nexusAllowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
    }

    // Get Nexus token balance
    function balanceOf(address user) public view returns (uint256) {
        return nexusBalances[user];
    }

    // Get Nexus token allowance
    function allowance(address user, address spender) public view returns (uint256) {
        return nexusAllowances[user][spender];
    }
}
