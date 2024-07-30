pragma solidity ^0.8.0;

contract SocialTradingContract {
    mapping (address => mapping (address => uint256)) public trades;

    function createTrade(address _token, uint256 _amount) public {
        require(_token != address(0), "Invalid token address");
        trades[msg.sender][_token] += _amount;
    }

    function getTrades(address _token) public view returns (uint256) {
        return trades[msg.sender][_token];
    }
}
