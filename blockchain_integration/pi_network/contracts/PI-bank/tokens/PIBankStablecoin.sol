pragma solidity ^0.8.0;

import "./IPIBankStablecoin.sol";

contract PIBankStablecoin is IPIBankStablecoin {
    mapping(address => uint256) public balances;
    uint256 public totalSupply;

    constructor() public {
        totalSupply = 1000000 ether; // 1 million stablecoins
        balances[msg.sender] = totalSupply;
    }

    function mint(address _owner, uint256 _amount) public {
        totalSupply += _amount;
        balances[_owner] += _amount;
    }

    function burn(uint256 _amount) public {
        totalSupply -= _amount;
        balances[msg.sender] -= _amount;
    }

    function transfer(address _from, address _to, uint256 _amount) public {
        balances[_from] -= _amount;
        balances[_to] += _amount;
    }

    function balanceOf(address _owner) public view returns (uint256) {
        return balances[_owner];
    }
}
