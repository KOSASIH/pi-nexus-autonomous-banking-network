pragma solidity ^0.8.0;

import "@openzeppelin/contracts/governance/Governor.sol";

contract Proposal {
    struct ProposalStruct {
        uint256 id;
        address proposer;
        string description;
        uint256 startBlock;
        uint256 endBlock;
        uint256 votesFor;
        uint256 votesAgainst;
        bool executed;
    }

    mapping(uint256 => ProposalStruct) public proposals;

    function propose(string memory description) external {
        //...
    }

    function vote(uint256 proposalId, bool support) external {
        //...
    }

    function execute(uint256 proposalId) external {
        //...
    }
}
