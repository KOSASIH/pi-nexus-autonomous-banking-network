// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract RoyaltyContract {
    mapping(address => uint256) public royalties;

    event RoyaltyPaid(address indexed creator, uint256 amount);

    function payRoyalty(address creator) external payable {
        require(msg.value > 0, "Royalty amount must be greater than zero");
        royalties[creator] += msg.value;
        emit RoyaltyPaid(creator, msg.value);
    }

    function withdrawRoyalties() external {
        uint256 amount = royalties[msg.sender];
        require(amount > 0, "No royalties to withdraw");
        royalties[msg.sender] = 0;
        payable(msg.sender).transfer(amount);
    }
}
