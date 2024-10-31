// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IVoting {
    // Events
    event ProposalCreated(uint256 indexed proposalId, string description, address indexed creator);
    event Voted(uint256 indexed proposalId, address indexed voter, bool support);

    // Structs
    struct Proposal {
        string description;
        uint256 voteCountFor;
        uint256 voteCountAgainst;
        uint256 endTime;
        bool executed;
    }

    // Functions
    function createProposal(string calldata description, uint256 duration) external returns (uint256);
    function vote(uint256 proposalId, bool support) external;
    function getProposal(uint256 proposalId) external view returns (Proposal memory);
    function hasVoted(uint256 proposalId, address voter) external view returns (bool);
}
