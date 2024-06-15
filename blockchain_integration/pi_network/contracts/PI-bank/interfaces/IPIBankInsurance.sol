pragma solidity ^0.8.4;

interface IPIBankInsurance {
    function createPolicy(uint256 policyId) external;
    function fileClaim(uint256 policyId, uint256 claimAmount) external;
    function getPolicy(address policyholder) external view returns (uint256);
    function getClaim(address policyholder, uint256 policyId) external view returns (uint256);
}
