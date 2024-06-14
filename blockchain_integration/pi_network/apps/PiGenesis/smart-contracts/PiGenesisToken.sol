pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract PiGenesisToken {
    string public name = "PiGenesis Token";
    string public symbol = "PGT";
    uint public totalSupply = 100000000 * (10 ** 18);

    mapping (address => uint) public balances;

    constructor() public {
        balances[msg.sender] = totalSupply;
    }

    function transfer(address _to, uint _value) public {
        require(balances[msg.sender] >= _value, "Insufficient balance");
        balances[msg.sender] -= _value;
        balances[_to] += _value;
    }

    function approve(address _spender, uint _value) public {
        // Implement approve logic
    }

    function transferFrom(address _from, address _to, uint _value) public {
        // Implement transferFrom logic
    }
}
