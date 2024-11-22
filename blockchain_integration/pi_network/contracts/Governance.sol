// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Governance is Ownable {
    IERC20 public token;

    struct Proposal {
        string description;
        uint256 voteCount;
        mapping(address => bool) voted;
        bool executed;
    uint256 proposalId;
    }

    mapping(uint256 => Proposal) public proposals;
    uint256 public proposalCount;

    event ProposalCreated(uint256 indexed proposalId, string description);
    event Voted(uint256 indexed proposalId, address indexed voter);
    event ProposalExecuted(uint256 indexed proposalId);

    constructor(IERC20 _token) {
        token = _token;
    }

    function createProposal(string memory description) external onlyOwner {
        proposalCount++;
        Proposal storage newProposal = proposals[proposalCount];
        newProposal.description = description;
        newProposal.proposalId = proposalCount;

        emit ProposalCreated(proposalCount, description);
    }

    function vote(uint256 proposalId) external {
        Proposal storage proposal = proposals[proposalId];
        require(!proposal.voted[msg.sender], "You have already voted");
        require(!proposal.executed, "Proposal already executed");

        uint256 voterBalance = token.balanceOf(msg.sender);
        require(voterBalance > 0, "You must own tokens to vote");

        proposal.voted[msg.sender] = true;
        proposal.voteCount += voterBalance;

        emit Voted(proposalId, msg.sender);
    }

    function executeProposal(uint256 proposalId) external {
        Proposal storage proposal = proposals[proposalId];
        require(!proposal.executed, "Proposal already executed");
        require(proposal.voteCount > (token.totalSupply() / 2), "Not enough votes");

        proposal.executed = true;

        // Implement the changes proposed (this is a placeholder)
        // For example, changing a contract address or updating a parameter

        emit ProposalExecuted(proposalId);
    }
}
