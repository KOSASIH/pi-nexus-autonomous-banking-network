pragma solidity ^0.8.0;

contract Oracle {
    address public sidraChainBridgeAddress;
    mapping (address => uint256) public prices;

    constructor(address _sidraChainBridgeAddress) public {
        sidraChainBridgeAddress = _sidraChainBridgeAddress;
    }

    function updatePrice(address _asset, uint256 _price) public {
        prices[_asset] = _price;
    }

    function getPrice(address _asset) public view returns (uint256) {
        return prices[_asset];
    }
}
