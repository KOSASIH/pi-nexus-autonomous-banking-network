pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract PiNexusBankingContract is ERC20 {
    address private _owner;
    uint256 private _totalSupply;

    constructor() public {
        _owner = msg.sender;
        _totalSupply = 100000000000; // Initial token supply
    }

    function mint(address _to, uint256 _amount) public onlyOwner {
        _mint(_to, _amount);
    }

    function burn(address _from, uint256 _amount) public onlyOwner {
        _burn(_from, _amount);
    }

    function transfer(address _to, uint256 _amount) public {
        _transfer(msg.sender, _to, _amount);
    }

    function approve(address _spender, uint256 _amount) public {
        _approve(msg.sender, _spender, _amount);
    }

    event Transfer(address indexed _from, address indexed _to, uint256 _amount);
    event Approval(address indexed _owner, address indexed _spender, uint256 _amount);
}
