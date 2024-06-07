pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract PiBridgeContract {
    address public piNetworkAddress;
    address public piTokenAddress;
    mapping (address => uint256) public userBalances;

    constructor(address _piNetworkAddress, address _piTokenAddress) public {
        piNetworkAddress = _piNetworkAddress;
        piTokenAddress = _piTokenAddress;
    }

    function deposit(uint256 amount) public {
        require(amount > 0, "Invalid deposit amount");
        SafeERC20.safeTransferFrom(piTokenAddress, msg.sender, address(this), amount);
        userBalances[msg.sender] += amount;
        emit Deposit(msg.sender, amount);
    }

    function withdraw(uint256 amount) public {
        require(amount > 0, "Invalid withdrawal amount");
        require(userBalances[msg.sender] >= amount, "Insufficient balance");
        SafeERC20.safeTransfer(piTokenAddress, msg.sender, amount);
        userBalances[msg.sender] -= amount;
        emit Withdrawal(msg.sender, amount);
    }

    event Deposit(address indexed user, uint256 amount);
    event Withdrawal(address indexed user, uint256 amount);
}
