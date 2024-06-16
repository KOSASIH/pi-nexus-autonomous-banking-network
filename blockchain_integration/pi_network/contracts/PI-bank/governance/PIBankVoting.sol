pragma solidity ^0.8.0;

import "./IPIBankVoting.sol";

contract PIBankVoting is IPIBankVoting {
    mapping(address => uint256) public votingPower;

    function castVote(address _voter, uint256 _proposalId, bool _vote) public {
        // implement voting logic
    }

    function countVotes(uint256 _proposalId) public view returns (uint256, uint256) {
        // implement vote counting logic
    }

    function determineOutcome(uint256 _proposalId) public view returns (bool){
        // implement outcome determination logic
    }
}
