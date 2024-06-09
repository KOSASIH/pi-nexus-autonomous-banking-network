pragma solidity ^0.8.0;

contract PiReentrancyDetector {
    uint256 public reentrancyCount;

    modifier nonReentrant() {
        require(reentrancyCount == 0, "Reentrancy detected");
        reentrancyCount++;
        _;
        reentrancyCount--;
    }
}
