pragma solidity ^0.8.4;

interface IPIBankGamification {
    function createChallenge(uint256 challengeId) external;
    function claimReward(uint256 challengeId, uint256 rewardAmount) external;
    function getChallenge(address challenger) external view returns (uint256);
    function getReward(address challenger, uint256 challengeId) external view returns (uint256);
}
