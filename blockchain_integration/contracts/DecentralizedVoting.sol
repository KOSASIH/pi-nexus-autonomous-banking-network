pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract DecentralizedVoting {
    // Mapping of voter addresses to voting power
    mapping (address => uint256) public votingPower;

    // Mapping of voter addresses to vote choices
    mapping (address => uint256) public votes;

    // Event emitted when a new vote is cast
    event VoteCast(address voter, uint256 choice);

    // Function to cast a new vote
    function castVote(uint256 _choice) public {
        // Check if voter has already voted
        require(votes[msg.sender] == 0, "Voter has already voted");

        // Check if voting power is valid
        require(votingPower[msg.sender] > 0, "Voter has no voting power");

        // Add vote to mapping
        votes[msg.sender] = _choice;

        // Emit vote cast event
        emit VoteCast(msg.sender, _choice);
    }

    // Function to calculatevote count for each choice
    function getVoteCount(uint256 _choice) public view returns (uint256) {
        uint256 voteCount = 0;

        // Iterate through all voters and count votes for each choice
        for (address voter : allVoters) {
            if (votes[voter] == _choice) {
                voteCount = voteCount.add(1);
            }
        }

        return voteCount;
    }

    // Function to get total voting power
    function getTotalVotingPower() public view returns (uint256) {
        uint256 totalVotingPower = 0;

        // Iterate through all voters and sum up voting power
        for (address voter : allVoters) {
            totalVotingPower = totalVotingPower.add(votingPower[voter]);
        }

        return totalVotingPower;
    }
}
