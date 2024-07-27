pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract KosasihUniversalisToken is ERC20 {
    string public name = "Kosasih Universalis Token";
    string public symbol = "KUT";
    uint256 public totalSupply = 1000000000000000000000000;

    constructor() public {
        // Initialize the token supply
        _mint(msg.sender, totalSupply);
    }
}
