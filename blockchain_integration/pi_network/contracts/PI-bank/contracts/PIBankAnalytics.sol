pragma solidity ^0.8.0;

import "./PIBank.sol";

contract PIBankAnalytics {
    // Mapping of analytics data
    mapping(address => AnalyticsData) public analyticsData;

    // Event
    event NewAnalyticsData(address indexed user, uint256 amount);

    // Function
    function updateAnalytics(address user, uint256 amount) public {
        // Update analytics data
        AnalyticsData data = AnalyticsData(user, amount);
        analyticsData[user] = data;
        emit NewAnalyticsData(user, amount);
    }
}
