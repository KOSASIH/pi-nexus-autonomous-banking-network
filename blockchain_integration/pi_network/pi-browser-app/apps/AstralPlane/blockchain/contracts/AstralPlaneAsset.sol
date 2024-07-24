// AstralPlaneAsset.sol
pragma solidity ^0.8.0;

contract AstralPlaneAsset {
    address private owner;
    mapping (address => uint256) public balances;

    constructor() public {
        owner = msg.sender;
    }

    function mint(address recipient, uint256 amount) public {
        require(msg.sender == owner, "Only the owner can mint assets");
        balances[recipient] += amount;
    }

    function transfer(address recipient, uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        balances[recipient] += amount;
    }

    function balanceOf(address account) public view returns (uint256) {
        return balances[account];
    }
}
