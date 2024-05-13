pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";

contract MultiChainBankManager is Ownable {
    IMultiChainBank[] public banks;

    function addBank(IMultiChainBank bank) external {
        require(msg.sender == owner, "Only the owner can add a bank");
        require(bank != address(0), "Invalid bank address");
        require(bank.totalSupply() > 0, "Bank has no tokens");
        require(bank.balanceOf(address(this)) == bank.totalSupply(), "Bank has insufficient balance");
        banks.push(bank);
    }

    function removeBank(IMultiChainBank bank) external {
        require(msg.sender == owner, "Only the owner can remove a bank");
        require(bank != address(0), "Invalid bank address");
        require(bank.totalSupply() > 0, "Bank has no tokens");
        require(bank.balanceOf(address(this)) == bank.totalSupply(), "Bank has insufficient balance");
        for (uint256 i = 0; i < banks.length; i++) {
            if (banks[i] == bank) {
                banks[i] = banks[banks.length - 1];
                banks.pop();
                break;
            }
        }
    }

    function getBanks() external view returns (IMultiChainBank[] memory) {
        return banks;
    }
}
