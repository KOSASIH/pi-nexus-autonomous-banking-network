pragma solidity ^0.8.4;

interface IPIBankYieldFarming {
    function stake(uint256 stakeAmount) external;
    function harvest() external;
    function getStake(address farmer) external view returns (uint256);
    function getReward(address farmer) external view returns (uint256);
}
