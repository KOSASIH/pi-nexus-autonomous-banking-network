// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Voting {
    struct Candidate {
        string name;
        uint256 voteCount;
    }

    mapping(uint256 => Candidate) public candidates;
    mapping(address => bool) public voters;
    uint256 public candidatesCount;
    uint256 public totalVotes;

    function addCandidate(string memory name) external {
        candidatesCount++;
        candidates[candidatesCount] = Candidate(name, 0);
    }

    function vote(uint256 candidateId) external {
        require(!voters[msg.sender], "You have already voted");
        require(candidateId > 0 && candidateId <= candidatesCount, "Invalid candidate ID");

        voters[msg.sender] = true;
        candidates[candidateId].voteCount++;
        totalVotes++;
    }

    function getResults() external view returns (string memory winnerName, uint256 winnerVoteCount) {
        uint256 winningVoteCount = 0;
for (uint256 i = 1; i <= candidatesCount; i++) {
            if (candidates[i].voteCount > winningVoteCount) {
                winningVoteCount = candidates[i].voteCount;
                winnerName = candidates[i].name;
            }
        }
        winnerVoteCount = winningVoteCount;
    }
}
