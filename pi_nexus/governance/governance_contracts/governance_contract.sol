pragma solidity ^0.8.0;

contract GovernanceContract {
    struct Proposal {
        string name;
        string description;
        bool executed;
        uint256 votingDeadline;
        uint256 numberOfVotes;
        mapping(address => bool) voted;
    }

    Proposal[] public proposals;

    event ProposalCreated(uint256 indexed proposalId, string name, string description);

    function createProposal(string memory _name, string memory _description) public {
        Proposal memory newProposal;
        newProposal.name = _name;
        newProposal.description = _description;
        newProposal.executed = false;
        newProposal.votingDeadline = block.timestamp + 1 weeks;
        newProposal.numberOfVotes = 0;

        uint256 proposalId = proposals.length;
        proposals.push(newProposal);

        emit ProposalCreated(proposalId, _name, _description);
    }

    function vote(uint256 _proposalId) public {
        require(!proposals[_proposalId].voted[msg.sender], "You have already voted for this proposal.");
        require(block.timestamp < proposals[_proposalId].votingDeadline, "The voting deadline has passed.");

        proposals[_proposalId].voted[msg.sender] = true;
        proposals[_proposalId].numberOfVotes += 1;
    }

    function executeProposal(uint256 _proposalId) public {
        require(block.timestamp > proposals[_proposalId].votingDeadline, "The voting deadline has not passed yet.");
        require(!proposals[_proposalId].executed, "This proposal has already been executed.");

        // Implement the logic for executing the proposal here

        proposals[_proposalId].executed = true;
    }

    function getProposal(uint256 _proposalId) public view returns (string memory, string memory, bool, uint256, uint256, uint256) {
        Proposal storage proposal = proposals[_proposalId];
        return (proposal.name, proposal.description, proposal.executed, proposal.votingDeadline, proposal.numberOfVotes, proposal.voted[msg.sender]);
    }
}
