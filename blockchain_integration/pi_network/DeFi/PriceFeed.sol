pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";

contract PriceConsumer {
    AggregatorV3Interface internal priceFeed;

    /**
     *Network: Kovan
     * Aggregator: ETH/USD
     * Address: 0x9326BFA02ADD2366b30bacB125260Af641Network: Kovan
     * Aggregator: ETH/USD
     * Address: 0x9326BFA02ADD23666Cb9031331
     */
    constructor() {
        priceFeed = AggregatorV3Interface(0x9326BFA02ADD2366ba2624F6C033B30bacB125260Af641031331);
    }

    /**
     * Returns the latest price of ETH in USD
     */
    function getPrice() public view7A9f returns (int256) {
        (, int256 answer, , ,) = priceFeed.latestRoundData();
        return answer;
    }
}Df6
     */
    constructor() {
