// genesis-smart-contracts/Waqf.sol
pragma solidity ^0.8.0;

contract Waqf {
    mapping (address => uint256) public balances;

    function receiveWaqf() public payable {
        balances[msg.sender] += msg.value;
    }

    function burnCoins(uint256 amount) public onlyOwner {
        // Burn coins
    }

    function getBalance() public view returns (uint256) {
        return address(this).balance;
    }
}
