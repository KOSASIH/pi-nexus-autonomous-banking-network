pragma solidity ^0.8.0;

contract PiRateLimiter {
    mapping (address => uint256) public requestCounts;
    uint256 public rateLimit;

    constructor(uint256 _rateLimit) public {
        rateLimit = _rateLimit;
    }

    modifier rateLimited() {
        require(requestCounts[msg.sender] < rateLimit, "Rate limit exceeded");
        requestCounts[msg.sender]++;
        _;
    }
}
