// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract GovernanceContract {
    IERC20 public governanceToken;
    mapping(address => uint256) public votes;
    mapping(uint256 => Proposal) public proposals;
    uint256 public proposalCount;

    struct Proposal {
        string description;
        uint256 voteCount;
        bool executed;
    }

    event ProposalCreated(uint256 proposalId, string description);
    event Voted(uint256 proposalId, address voter, uint256 votes);
    event ProposalExecuted(uint256 proposalId);

    constructor(IERC20 _governanceToken) {
        governanceToken = _governanceToken;
    }

    function createProposal(string memory description) external {
        proposalCount++;
        proposals[proposalCount] = Proposal(description, 0, false);
        emit ProposalCreated(proposalCount, description);
    }

    function vote(uint256 proposalId, uint256 amount) external {
        require(amount > 0, "Must vote with a positive amount");
        require(governanceToken.transferFrom(msg.sender, address(this), amount), "Transfer failed");

        proposals[proposalId].voteCount += amount;
        votes[msg.sender] += amount;

        emit Voted(proposalId, msg.sender, amount);
    }

    function executeProposal(uint256 proposalId) external {
        Proposal storage proposal = proposals[proposalId];
        require(!proposal.executed, "Proposal already executed");
        require(proposal.voteCount > 0, "No votes for this proposal");

        proposal.executed = true;
        emit ProposalExecuted(proposalId);
        // Execute proposal logic here
    }
}
