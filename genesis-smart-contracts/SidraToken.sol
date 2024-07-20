// genesis-smart-contracts/SidraToken.sol
pragma solidity ^0.8.0;

contract SidraToken {
    string public symbol;
    string public name;
    uint256 public decimals;
    uint256 public totalSupply;
    uint256 public circulatingSupply;

    mapping (address => uint256) public balances;

    function mint(address recipient, uint256 amount) public onlyOwner {
        balances[recipient] += amount;
        totalSupply += amount;
        circulatingSupply += amount;
    }

    function burn(uint256 amount) public onlyOwner {
        totalSupply -= amount;
        circulatingSupply -= amount;
    }

    function pause() public onlyOwner {
        // Pause token operations
    }

    function unpause() public onlyOwner {
        // Unpause token operations
    }
}
