// dao.sol
pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Roles.sol";

contract DAO {
    using Roles for address;

    struct Proposal {
        address proposer;
        string description;
        uint256 votingDeadline;
        uint256 yesVotes;
        uint256 noVotes;
    }

    mapping (address => Proposal[]) public proposals;
    mapping (address => uint256) public votingPower;

    eventNewProposal(address indexed proposer, string description);
    eventVoteCast(address indexed voter, uint256 proposalId, bool in Favor);

    function createProposal(string memory _description) public {
        require(msg.sender != address(0), "Invalid sender");
        Proposal memory proposal = Proposal(msg.sender, _description, block.timestamp + 30 days, 0, 0);
        proposals[msg.sender].push(proposal);
        emit NewProposal(msg.sender, _description);
    }

    function vote(uint256 _proposalId, bool _inFavor) public {
        require(msg.sender != address(0), "Invalid sender");
        Proposal storage proposal = proposals[msg.sender][_proposalId];
        require(proposal.votingDeadline > block.timestamp, "Voting deadline has passed");
        if (_inFavor) {
            proposal.yesVotes += votingPower[msg.sender];
        } else {
            proposal.noVotes += votingPower[msg.sender];
        }
        emit VoteCast(msg.sender, _proposalId, _inFavor);
    }

    function executeProposal(uint256 _proposalId) public {
        require(msg.sender != address(0), "Invalid sender");
        Proposal storage proposal = proposals[msg.sender][_proposalId];
        require(proposal.votingDeadline < block.timestamp, "Voting deadline has not passed");
        if (proposal.yesVotes > proposal.noVotes) {
            // Execute proposal
        } else {
            // Reject proposal
        }
    }
}
