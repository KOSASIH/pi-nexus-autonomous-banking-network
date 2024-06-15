pragma solidity ^0.8.4;

interface IPIBankGovernance {
    function propose(uint256 proposalId) external;
    function vote(uint256 proposalId, uint256 vote) external;
    function getProposal(address proposer) external view returns (uint256);
    function getVote(address voter, uint256 proposalId) external view returns (uint256);
}
