pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract NexusToken is ERC20 {
    address private _owner;
    uint256 private _totalSupply;

    constructor() public {
        _owner = msg.sender;
        _totalSupply = 1000000 * (10 ** 18); // 1 million tokens with 18 decimals
        _mint(_owner, _totalSupply);
    }

    function transfer(address recipient, uint256 amount) public override {
        require(recipient != address(0), "Cannot transfer to zero address");
        _transfer(msg.sender, recipient, amount);
    }

    function _transfer(address sender, address recipient, uint256 amount) internal {
        require(sender != address(0), "Cannot transfer from zero address");
        require(recipient != address(0), "Cannot transfer to zero address");
        _balances[sender] = _balances[sender].sub(amount, "Insufficient balance");
        _balances[recipient] = _balances[recipient].add(amount);
        emit Transfer(sender, recipient, amount);
    }
}
