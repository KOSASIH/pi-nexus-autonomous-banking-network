pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract OptimizedPiToken is ERC20 {
    constructor(uint256 initialSupply) ERC20("Optimized Pi Token", "OPTPI") {
        _mint(msg.sender, initialSupply);
    }

    function transfer(address recipient, uint256 amount) public returns (bool) {
        _transfer(msg.sender, recipient, amount);
        return true;
    }

    function approve(address spender, uint256 amount) public returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public returns (bool) {
        _transfer(sender, recipient, amount);
        _approve(sender, msg.sender, _balanceOf(sender).sub(amount));
        return true;
    }
}
