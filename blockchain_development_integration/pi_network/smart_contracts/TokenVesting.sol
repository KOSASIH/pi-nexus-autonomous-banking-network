// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract TokenVesting {
    IERC20 public token;
    address public beneficiary;
    uint256 public start;
    uint256 public cliff;
    uint256 public duration;
    uint256 public released;

    constructor(
        IERC20 _token,
        address _beneficiary,
        uint256 _start,
        uint256 _cliff,
        uint256 _duration
    ) {
        require(_beneficiary != address(0), "Beneficiary is the zero address");
        require(_cliff >= _start, "Cliff is before start");
        require(_duration > 0, "Duration is 0");

        token = _token;
        beneficiary = _beneficiary;
        start = _start;
        cliff = _cliff;
        duration = _duration;
    }

    function release() external {
        require(block.timestamp >= cliff, "Cliff not reached");
        uint256 unreleased = _releasableAmount();
        require(unreleased > 0, "No tokens to release");

        released += unreleased;
        token.transfer(beneficiary, unreleased);
    }

    function _releasableAmount() internal view returns (uint256) {
        if (block.timestamp < start) {
            return 0;
        } else if (block.timestamp >= start + duration) {
            return token.balanceOf(address(this));
        } else {
            return (token.balanceOf(address(this)) * (block.timestamp - start)) / duration;
        }
    }

    function getVestedAmount() external view returns (uint256) {
        return _releasableAmount();
    }
}
