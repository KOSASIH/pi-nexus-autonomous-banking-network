pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract PIWallet {
    using SafeMath for uint256;

    address public owner;
    mapping (address => uint256) public balances;

    event Deposit(address indexed user, uint256 amount);
    event Withdrawal(address indexed user, uint256 amount);

    constructor() public {
        owner = msg.sender;
    }

    function deposit() public payable {
        balances[msg.sender] = balances[msg.sender].add(msg.value);
        emit Deposit(msg.sender, msg.value);
    }

    function withdraw(uint256 _amount) public {
        require(_amount <= balances[msg.sender], "Insufficient balance");
        msg.sender.transfer(_amount);
        balances[msg.sender] = balances[msg.sender].sub(_amount);
        emit Withdrawal(msg.sender, _amount);
    }

    function transfer(address _to, uint256 _amount) public {
        require(_amount <= balances[msg.sender], "Insufficient balance");
        balances[msg.sender] = balances[msg.sender].sub(_amount);
        balances[_to] = balances[_to].add(_amount);
        emit Transfer(msg.sender, _to, _amount);
    }

    function withdrawTo(address _to, uint256 _amount) public onlyOwner {
        require(_to != address(0), "Invalid address");
        require(_amount > 0, "Invalid amount");

        _to.transfer(_amount);

        emit Withdrawal(msg.sender, _to, _amount);
     }
}
