pragma solidity ^0.8.0;

interface IMultiChainBankManager {
    function addBank(IMultiChainBank bank) external;
    function removeBank(IMultiChainBank bank) external;
    function getBanks() external view returns (IMultiChainBank[] memory);
}
