pragma solidity ^0.8.4;

interface IPIBankAnalytics {
    function getTotalValueLocked() external view returns (uint256);
    function getVolume(uint256 startTime, uint256 endTime) external view returns (uint256);
    function getLiquidity(uint256 startTime, uint256 endTime) external view returns (uint256);
}
