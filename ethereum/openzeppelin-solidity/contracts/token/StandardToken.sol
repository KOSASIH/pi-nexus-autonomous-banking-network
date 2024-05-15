// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract StandardToken is ERC20 {
    constructor(uint256 initialSupply, string memory name, string memory symbol) ERC20(name, symbol) {
        _mint(msg.sender, initialSupply);
    }

    function mint(address to, uint256 amount) public onlyOwner {
        _mint(to, amount);
    }

    function burn(uint256 amount) public onlyOwner {
        _burn(msg.sender, amount);
    }

    function burnFrom(address from, uint256 amount) public {
        _burn(from, amount);
    }

    function transferAnyERC20(address token, address to, uint256 amount) public {
        IERC20Upgradeable(token).transferFrom(msg.sender, address(this), amount);
        _transfer(address(this), to, amount);
    }

    function totalBurned() public view returns (uint256) {
        return _totalBurned();
    }
}
