// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract GovernanceVotingSystem {
    struct Proposal {
        uint id;
        string description;
        uint voteCount;
        uint endTime;
        bool executed;
        mapping(address => bool) voters;
    }

    mapping(uint => Proposal) public proposals;
    mapping(address => uint) public votingPower;
    uint public proposalCount;
    uint public votingDuration;

    event ProposalCreated(uint id, string description, uint endTime);
    event Voted(uint proposalId, address voter);
    event ProposalExecuted(uint proposalId);

    constructor(uint _votingDuration) {
        votingDuration = _votingDuration; // Duration for voting in seconds
    }

    // Function to create a new proposal
    function createProposal(string memory description) public {
        proposalCount++;
        uint endTime = block.timestamp + votingDuration;

        Proposal storage newProposal = proposals[proposalCount];
        newProposal.id = proposalCount;
        newProposal.description = description;
        newProposal.endTime = endTime;

        emit ProposalCreated(proposalCount, description, endTime);
    }

    // Function to vote on a proposal
    function vote(uint proposalId) public {
        Proposal storage proposal = proposals[proposalId];

        require(block.timestamp < proposal.endTime, "Voting has ended");
        require(!proposal.voters[msg.sender], "You have already voted");

        proposal.voters[msg.sender] = true;
        proposal.voteCount += votingPower[msg.sender];

        emit Voted(proposalId, msg.sender);
    }

    // Function to execute a proposal if it has enough votes
    function executeProposal(uint proposalId) public {
        Proposal storage proposal = proposals[proposalId];

        require(block.timestamp >= proposal.endTime, "Voting is still ongoing");
        require(!proposal.executed, "Proposal has already been executed");
        require(proposal.voteCount > 0, "No votes cast");

        proposal.executed = true;

        // Logic to execute the proposal goes here
        // For example, changing a state variable or transferring funds

        emit ProposalExecuted(proposalId);
    }

    // Function to set voting power for an address (for demonstration purposes)
    function setVotingPower(address voter, uint power) public {
        votingPower[voter] = power;
    }
}
