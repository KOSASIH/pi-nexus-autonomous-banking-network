pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/Governance.sol";

contract PiNetworkGovernance is Governance {
    // Mapping of proposals to their corresponding votes
    mapping (bytes32 => mapping (address => uint256)) public proposalVotes;

    // Event emitted when a new proposal is created
    event ProposalCreatedEvent(bytes32 indexed proposal, address indexed creator);

    // Function to create a new proposal
    function createProposal(bytes32 proposal, string memory description) public {
        proposalVotes[proposal][msg.sender] = 1;
        emit ProposalCreatedEvent(proposal, msg.sender);
    }

    // Function to vote on a proposal
    function voteOnProposal(bytes32 proposal, uint256 vote) public {
        proposalVotes[proposal][msg.sender] = vote;
    }

    // Function to get the result of a proposal
    function getProposalResult(bytes32 proposal) public view returns (uint256) {
        uint256 yesVotes = 0;
        uint256 noVotes = 0;
        for (address voter in proposalVotes[proposal]) {
            if (proposalVotes[proposal][voter] == 1) {
                yesVotes++;
            } else {
                noVotes++;
            }
        }
        return yesVotes > noVotes? 1 : 0;
    }
}
