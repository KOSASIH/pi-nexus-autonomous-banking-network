pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract ERC20Factory {
    mapping (address => ERC20) public tokens;

    function createToken(string memory _name, string memory _symbol, uint256 _totalSupply) public {
        ERC20 token = new ERC20(_name, _symbol, _totalSupply);
        tokens[msg.sender] = token;
    }

    function getToken(address _owner) public view returns (ERC20) {
        return tokens[_owner];
    }
}
