// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts-upgradeable/token/ERC20/IERC20Upgradeable.sol";
import "@openzeppelin/contracts-upgradeable/token/ERC20/utils/SafeERC20Upgradeable.sol";
import "@openzeppelin/contracts-upgradeable/security/PausableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";

contract FantasyExchange is
  PausableUpgradeable,
  OwnableUpgradeable,
  IERC20Upgradeable
{
  using SafeERC20Upgradeable for IERC20Upgradeable;

  AggregatorV3Interface public priceFeed;

  uint256 public constant MIN_ORDER_VALUE = 1 ether;

  struct Order {
    IERC20Upgradeable token;
    address taker;
    uint256 takerAmount;
    uint256 makerAmount;
    uint256 fee;
    uint8 v;
    bytes32 r;
    bytes32 s;
  }

  struct Market {
    IERC20Upgradeable token;
    uint256 basePrice;
    uint256 quoteIncrement;
  }

  Market[] public markets;

  event OrderCreated(
    uint256 indexed id,
    address indexed maker,
    address indexed taker,
    uint256 makerAmount,
    uint256 takerAmount,
    uint256 fee,
    uint8 v,
    bytes32 r,
    bytes32 s
  );

  event OrderFilled(
    uint256 indexed id,
    address indexed maker,
    address indexed taker,
    uint256 makerAmount,
    uint256 takerAmount,
    uint256 fee,
    uint8 v,
    bytes32 r,
    bytes32 s
  );

  constructor(address priceFeedAddress) {
    priceFeed = AggregatorV3Interface(priceFeedAddress);
  }

  function createMarket(IERC20Upgradeable token) external onlyOwner {
    markets.push(Market({
      token: token,
      basePrice: 0,
      quoteIncrement: 1
    }));
  }

  function createOrder(
    uint256 marketIndex,
    uint256 takerAmount,
    uint256 makerAmount,
    uint256 fee
  ) external {
    Market storage market = markets[marketIndex];
    require(market.token.safeTransferFrom(msg.sender, address(this), takerAmount), "Transfer failed");
    uint256 id = uint256(keccak256(abi.encodePacked(msg.sender, block.timestamp)));
    Order memory order = Order({
      token: market.token,
      taker: msg.sender,
      takerAmount: takerAmount,
      makerAmount: makerAmount,
      fee: fee,
      v: 27,
      r: bytes32(0),
      s: bytes32(0)
    });
    emit OrderCreated(id, msg.sender, address(0), makerAmount, takerAmount, fee, 27, bytes32(0), bytes32(0));
    fillOrder(id, order);
  }

  function fillOrder(uint256 id, Order memory order) external {
    require(msg.sender != order.taker, "Cannot fill your own order");
    require(block.timestamp <= order.taker.deadline, "Order expired");
    require(order.takerAmount >= order.fee, "Insufficient taker amount");
    require(order.takerAmount >= order.makerAmount, "Insufficient taker amount");
    require(order.token.balanceOf(address(this)) >= order.makerAmount, "Insufficient balance");
    uint256 price = order.takerAmount.div(order.makerAmount);
    require(price >= market.basePrice.div(market.quoteIncrement), "Price too low");
    require(price <= market.basePrice.mul(market.quoteIncrement), "Price too high");
    require(order.token.safeTransfer(order.taker, order.makerAmount), "Transfer failed");
    require(order.token.safeTransfer(address(this), order.takerAmount), "Transfer failed");
    emit OrderFilled(id, msg.sender, order.taker, order.makerAmount, order.takerAmount, order.fee, 27, bytes32(0), bytes32(0));
  }

  function getPrice() external view returns (uint256) {
    (, int256 answer, , ,) = priceFeed.latestRoundData();
    require(answer > 0, "Invalid price");
    return uint256(answer);
  }
}
