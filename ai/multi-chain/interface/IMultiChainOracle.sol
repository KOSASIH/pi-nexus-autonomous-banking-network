pragma solidity ^0.8.0;

interface IMultiChainOracle {
    function getPrice(address token) external view returns (uint256);
}
