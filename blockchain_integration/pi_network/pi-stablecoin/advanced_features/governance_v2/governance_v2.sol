pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract GovernanceV2 {
    // Mapping of proposals
    mapping (uint256 => Proposal) public proposals;

    // Function to create a new proposal
    function createProposal(string memory description, address[] memory targets, uint256[] memory values) public {
        // Create a new proposal
        Proposal proposal = Proposal(description, targets, values);
        proposals[proposalId] = proposal;
        // Emit event to notify voters
        emit NewProposal(proposalId);
    }
}
