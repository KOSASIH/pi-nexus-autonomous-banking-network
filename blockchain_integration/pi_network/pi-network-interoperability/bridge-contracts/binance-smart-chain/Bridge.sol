pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/binance-chain/bsc-bridge/blob/master/contracts/BEP20.sol";

contract Bridge {
    address public owner;
    mapping(address => uint256) public balances;
    mapping(address => mapping(address => uint256)) public allowances;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor() public {
        owner = msg.sender;
    }

    function bridgeToken(address tokenAddress, address recipientAddress, uint256 amount) public {
        require(msg.sender == owner, "Only the owner can bridge tokens");
        SafeERC20.safeTransfer(tokenAddress, recipientAddress, amount);
        balances[tokenAddress] -= amount;
    }

    function getBalance(address tokenAddress) public view returns (uint256) {
        return balances[tokenAddress];
    }

    function approve(address spender, uint256 amount) public {
        allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
    }

    function transferFrom(address sender, address recipient, uint256 amount) public {
        require(allowances[sender][msg.sender] >= amount, "Insufficient allowance");
        SafeERC20.safeTransfer(sender, recipient, amount);
        allowances[sender][msg.sender] -= amount;
    }
}
