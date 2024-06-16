pragma solidity ^0.8.0;

import "./IPIBankGovernance.sol";

contract PIBankGovernance is IPIBankGovernance {
    mapping(address => uint256) public votingPower;
    mapping(uint256 => Proposal) public proposals;
    uint256 public proposalCount;

    struct Proposal {
        string description;
        address proposer;
        uint256 votingDeadline;
        uint256 yesVotes;
        uint256 noVotes;
    }

    function propose(address _proposer, string calldata _description) public {
        Proposal memory proposal = Proposal(_description, _proposer, block.timestamp + 30 days, 0, 0);
        proposals[proposalCount] = proposal;
        proposalCount++;
    }

    function vote(address _voter, uint256 _proposalId, bool _vote) public {
        Proposal storage proposal = proposals[_proposalId];
        if (_vote) {
            proposal.yesVotes += votingPower[_voter];
        } else {
            proposal.noVotes += votingPower[_voter];
        }
    }

    function executeProposal(uint256 _proposalId) public {
        Proposal storage proposal = proposals[_proposalId];
        if (proposal.yesVotes > proposal.noVotes) {
            // execute proposal
        }
    }

    function getProposal(uint256 _proposalId) public view returns (string memory, address, uint256) {
        Proposal storage proposal = proposals[_proposalId];
        return (proposal.description, proposal.proposer, proposal.votingDeadline);
    }

    function getVotingPower(address _voter) public view returns (uint256) {
        return votingPower[_voter];
    }
}
