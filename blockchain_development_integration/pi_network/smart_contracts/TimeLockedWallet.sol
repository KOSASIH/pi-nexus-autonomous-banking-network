// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TimeLockedWallet {
    address public owner;
    uint256 public unlockTime;

    constructor(uint256 _lockDuration) {
        owner = msg.sender;
        unlockTime = block.timestamp + _lockDuration;
    }

    function deposit() external payable {
        require(msg.value > 0, "Must send Ether");
    }

    function withdraw() external {
        require(msg.sender == owner, "Not the owner");
        require(block.timestamp >= unlockTime, "Funds are locked");

        payable(owner).transfer(address(this).balance);
    }

    function getLockedFunds() external view returns (uint256) {
        return address(this).balance;
    }
}
