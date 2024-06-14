pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "./PiGenesisToken.sol";

contract PortfolioManager {
    address public owner;
    mapping (address => uint) public portfolioBalances;

    constructor() public {
        owner = msg.sender;
    }

    function deposit(address _token, uint _value) public {
        require(msg.sender == owner, "Only owner can deposit");
        PiGenesisToken token = PiGenesisToken(_token);
        token.transferFrom(msg.sender, address(this), _value);
        portfolioBalances[msg.sender] += _value;
    }

    function withdraw(address _token, uint _value) public {
        require(msg.sender == owner, "Only owner can withdraw");
        PiGenesisToken token = PiGenesisToken(_token);
        token.transfer(msg.sender, _value);
        portfolioBalances[msg.sender] -= _value;
    }
}
