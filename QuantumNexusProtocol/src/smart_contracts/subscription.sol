// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SubscriptionContract {
    struct Subscription {
        uint256 amount;
        uint256 interval;
        uint256 lastPayment;
        bool isActive;
    }

    mapping(address => Subscription) public subscriptions;

    event SubscriptionCreated(address indexed subscriber, uint256 amount, uint256 interval);
    event SubscriptionCancelled(address indexed subscriber);

    function createSubscription(uint256 amount, uint256 interval) external {
        subscriptions[msg.sender] = Subscription(amount, interval, block.timestamp, true);
        emit SubscriptionCreated(msg.sender, amount, interval);
    }

    function paySubscription() external payable {
        Subscription storage subscription = subscriptions[msg.sender];
        require(subscription.isActive, "No active subscription");
        require(msg.value == subscription.amount, "Incorrect payment amount");
        require(block.timestamp >= subscription.lastPayment + subscription.interval, "Payment not due");

        subscription.lastPayment = block.timestamp;
    }

    function cancelSubscription() external {
        Subscription storage subscription = subscriptions[msg.sender];
        require(subscription.isActive, "No active subscription");
        subscription.isActive = false;
        emit SubscriptionCancelled(msg.sender);
    }
}
