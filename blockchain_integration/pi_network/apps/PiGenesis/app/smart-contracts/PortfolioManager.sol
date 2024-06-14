pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/ownership/Ownable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "./PiGenesisToken.sol";

contract PortfolioManager {
    using SafeMath for uint256;

    address public owner;
    PiGenesisToken public pgtToken;

    mapping (address => uint256) public portfolioBalances;
    mapping (address => mapping (address => uint256)) public portfolioAllowances;

    event Deposit(address indexed user, uint256 amount);
    event Withdrawal(address indexed user, uint256 amount);
    event Transfer(address indexed from, address indexed to, uint256 amount);

    constructor() public {
        owner = msg.sender;
        pgtToken = PiGenesisToken(address(new PiGenesisToken()));
    }

    function deposit(uint256 _amount) public {
        require(_amount > 0, "Invalid deposit amount");
        portfolioBalances[msg.sender] = portfolioBalances[msg.sender].add(_amount);
        pgtToken.transferFrom(msg.sender, address(this), _amount);
        emit Deposit(msg.sender, _amount);
    }

    function withdraw(uint256 _amount) public {
        require(_amount > 0, "Invalid withdrawal amount");
        require(portfolioBalances[msg.sender] >= _amount, "Insufficient balance");
        portfolioBalances[msg.sender] = portfolioBalances[msg.sender].sub(_amount);
        pgtToken.transfer(msg.sender, _amount);
        emit Withdrawal(msg.sender, _amount);
    }

    function transfer(address _to, uint256 _amount) public {
        require(_amount > 0, "Invalid transfer amount");
        require(portfolioBalances[msg.sender] >= _amount, "Insufficient balance");
        portfolioBalances[msg.sender] = portfolioBalances[msg.sender].sub(_amount);
        portfolioBalances[_to] = portfolioBalances[_to].add(_amount);
        emit Transfer(msg.sender, _to, _amount);
    }
}
