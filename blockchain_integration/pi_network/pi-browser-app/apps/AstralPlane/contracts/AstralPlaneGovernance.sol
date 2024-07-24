pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/governance/Governor.sol";

contract AstralPlaneGovernance is Governor {
    struct Proposal {
        address proposer;
        string description;
        uint256 votingPeriod;
        uint256 executionDelay;
        bytes[] targets;
        uint256[] values;
        bytes[] calldatas;
    }

    mapping (uint256 => Proposal) public proposals;

    event ProposalCreated(uint256 proposalId, address proposer, string description);
    event VoteCast(address voter, uint256 proposalId, bool support);
    event ProposalExecuted(uint256 proposalId);

    function propose(address[] memory targets, uint256[] memory values, bytes[] memory calldatas, string memory description) public {
        uint256 proposalId = uint256(keccak256(abi.encodePacked(targets, values, calldatas, description)));
        proposals[proposalId] = Proposal(msg.sender, description, block.timestamp + 7 days, block.timestamp + 14 days, targets, values, calldatas);
        emit ProposalCreated(proposalId, msg.sender, description);
    }

    function vote(uint256 proposalId, bool support) public {
        require(proposals[proposalId].votingPeriod > block.timestamp, "Voting period has ended");
        emit VoteCast(msg.sender, proposalId, support);
    }

    function execute(uint256 proposalId) public {
        require(proposals[proposalId].executionDelay <= block.timestamp, "Execution delay has not passed");
        for (uint256 i = 0; i < proposals[proposalId].targets.length; i++) {
            (bool success, ) = proposals[proposalId].targets[i].call.value(proposals[proposalId].values[i])(proposals[proposalId].calldatas[i]);
            require(success, "Execution failed");
        }
        emit ProposalExecuted(proposalId);
    }
}
