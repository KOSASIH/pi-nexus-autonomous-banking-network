// governance_smart_contract.sol
pragma solidity ^0.8.0;

contract Governance {
    address[] public stakeholders;
    mapping (address => uint256) public votes;

    function propose(address proposer, string memory proposal) public {
        // validate proposer and proposal
        stakeholders.push(proposer);
        votes[proposer] = 0;
    }

    function vote(address voter, address proposer) public {
        // validate voter and proposer
        votes[proposer]++;
    }

    function executeProposal(address proposer) public {
        // validate proposer and proposal
        if (votes[proposer] > stakeholders.length / 2) {
            // execute proposal
        }
    }
}
