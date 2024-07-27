pragma solidity ^0.8.0;

interface IMultiChainRouter {
    function transferTokens(address _token, address _from, address _to, uint256 _amount) external;
}
