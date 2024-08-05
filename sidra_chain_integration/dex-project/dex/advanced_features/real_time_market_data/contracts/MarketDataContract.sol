pragma solidity ^0.8.0;

contract MarketDataContract {
    mapping (address => mapping (address => uint256)) public marketData;

    function updateMarketData(address _token, uint256 _price) public {
        require(_token != address(0), "Invalid token address");
        marketData[msg.sender][_token] = _price;
    }

    function getMarketData(address _token) public view returns (uint256) {
        return marketData[msg.sender][_token];
    }
}
