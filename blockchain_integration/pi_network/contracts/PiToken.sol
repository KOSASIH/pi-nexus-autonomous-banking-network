pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/ownership/Ownable.sol";

contract PiToken is ERC20, Ownable {
    string public name = "Pi Token";
    string public symbol = "PI";
    uint256 public totalSupply = 100000000 * (10**18);

    mapping (address => uint256) public balances;

    constructor() public {
        balances[msg.sender] = totalSupply;
    }

    function transfer(address _to, uint256 _value) public {
        require(balances[msg.sender] >= _value, "Insufficient balance");
        balances[msg.sender] -= _value;
        balances[_to] += _value;
    }

    function burn(uint256 _value) public onlyOwner {
        require(balances[msg.sender] >= _value, "Insufficient balance");
        balances[msg.sender] -= _value;
        totalSupply -= _value;
    }

    function mint(uint256 _value) public onlyOwner {
        totalSupply += _value;
        balances[msg.sender] += _value;
    }
}
