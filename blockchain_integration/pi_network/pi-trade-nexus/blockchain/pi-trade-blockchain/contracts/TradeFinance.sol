pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "./PiTradeToken.sol";

contract TradeFinance is SafeERC20 {
    PiTradeToken public piTradeToken;
    mapping (address => mapping (address => uint256)) public tradeBalances;
    mapping (address => mapping (address => uint256)) public tradeAllowances;

    event TradeInitiated(address indexed buyer, address indexed seller, uint256 value);
    event TradeConfirmed(address indexed buyer, address indexed seller, uint256 value);
    event TradeCancelled(address indexed buyer, address indexed seller, uint256 value);

    constructor(address _piTradeToken) public {
        piTradeToken = PiTradeToken(_piTradeToken);
    }

    function initiateTrade(address _seller, uint256 _value) public returns (bool) {
        require(_seller != address(0));
        require(_value > 0);

        tradeBalances[msg.sender][_seller] = _value;
        emit TradeInitiated(msg.sender, _seller, _value);
        return true;
    }

    function confirmTrade(address _buyer, uint256 _value) public returns (bool) {
        require(_buyer != address(0));
        require(_value > 0);
        require(tradeBalances[_buyer][msg.sender] == _value);

        piTradeToken.transferFrom(_buyer, msg.sender, _value);
        tradeAllowances[_buyer][msg.sender] = _value;
        emit TradeConfirmed(_buyer, msg.sender, _value);
        return true;
    }

    function cancelTrade(address _seller, uint256 _value) public returns (bool) {
        require(_seller != address(0));
        require(_value > 0);
        require(tradeBalances[msg.sender][_seller] == _value);

        tradeBalances[msg.sender][_seller] = 0;
        emit TradeCancelled(msg.sender, _seller, _value);
        return true;
    }

    function getTradeBalance(address _buyer, address _seller) public view returns (uint256) {
        return tradeBalances[_buyer][_seller];
    }

    function getTradeAllowance(address _buyer, address _seller) public view returns (uint256) {
        return tradeAllowances[_buyer][_seller];
    }
}
