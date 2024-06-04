pragma solidity ^0.8.0;

contract Governance {
    struct Proposal {
        string description;
        bool executed;
    }

    Proposal[] public proposals;

    function createProposal(string memory _description) public {
        Proposal memory newProposal = Proposal({
            description: _description,
            executed: false
        });

        proposals.push(newProposal);
    }

    function executeProposal(uint _index) public {
        Proposal storage proposal = proposals[_index];

        if (!proposal.executed) {
            // Implement the logic for executing the proposal here
            proposal.executed = true;
        }
    }
}
