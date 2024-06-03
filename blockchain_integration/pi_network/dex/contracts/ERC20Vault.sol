pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract ERC20Vault is Ownable {
    function deposit(IERC20 token, uint256 amount) public onlyOwner {
        token.transferFrom(msg.sender, address(this), amount);
    }

    function withdraw(IERC20 token, uint256 amount) public onlyOwner {
        token.transfer(msg.sender, amount);
    }
}
