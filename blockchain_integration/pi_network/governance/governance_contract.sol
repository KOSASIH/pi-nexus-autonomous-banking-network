// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Governance {
    event ProposalCreated(uint indexed proposalId, string description, address indexed proposer);
    event Voted(uint indexed proposalId, address indexed voter, bool support);
    event ProposalExecuted(uint indexed proposalId);

    struct Proposal {
        string description;
        address proposer;
        uint voteCountFor;
        uint voteCountAgainst;
        mapping(address => bool) hasVoted;
        bool executed;
    }

    Proposal[] public proposals;
    uint public votingPeriod;
    address public owner;

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the contract owner");
        _;
    }

    constructor(uint _votingPeriod) {
        owner = msg.sender;
        votingPeriod = _votingPeriod;
    }

    function createProposal(string memory description) public {
        Proposal storage newProposal = proposals.push();
        newProposal.description = description;
        newProposal.proposer = msg.sender;
        emit ProposalCreated(proposals.length - 1, description, msg.sender);
    }

    function vote(uint proposalId, bool support) public {
        Proposal storage proposal = proposals[proposalId];
        require(!proposal.hasVoted[msg.sender], "Already voted");
        require(!proposal.executed, "Proposal already executed");

        if (support) {
            proposal.voteCountFor++;
        } else {
            proposal.voteCountAgainst++;
        }

        proposal.hasVoted[msg.sender] = true;
        emit Voted(proposalId, msg.sender, support);
    }

    function executeProposal(uint proposalId) public {
        Proposal storage proposal = proposals[proposalId];
        require(!proposal.executed, "Proposal already executed");
        require(block.timestamp >= votingPeriod, "Voting period not ended");

        if (proposal.voteCountFor > proposal.voteCountAgainst) {
            // Execute the proposal (custom logic can be added here)
            proposal.executed = true;
            emit ProposalExecuted(proposalId);
        } else {
            revert("Proposal not approved");
        }
    }

    function getProposal(uint proposalId) public view returns (string memory, address, uint, uint, bool) {
        Proposal storage proposal = proposals[proposalId];
        return (proposal.description, proposal.proposer, proposal.voteCountFor, proposal.voteCountAgainst, proposal.executed);
    }

    function getProposalCount() public view returns (uint) {
        return proposals.length;
    }
}
