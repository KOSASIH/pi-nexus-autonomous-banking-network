// genesis-smart-contracts/MainFaucet.sol
pragma solidity ^0.8.0;

contract MainFaucet {
    mapping (address => uint256) public balances;

    function distributeCoins(address recipient, uint256 amount) public onlyOwner {
        balances[recipient] += amount;
    }

    function getBalance() public view returns (uint256) {
        return address(this).balance;
    }
}
