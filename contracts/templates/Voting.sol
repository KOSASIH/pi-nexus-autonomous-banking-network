pragma solidity ^0.8.0;

contract Voting {

    // The list of candidates
    struct Candidate {
        string name;
        uint voteCount;
    }
    Candidate[] public candidates;

    // The function to initialize the contract
    constructor() {
        addCandidate('Candidate 1');
        addCandidate('Candidate 2');
        addCandidate('Candidate 3');
    }

    // The function to add a candidate
    function addCandidate(string memory _name) private {
        candidates.push(Candidate({
            name: _name,
            voteCount: 0
        }));
    }

    // The function to vote for a candidate
    function vote(uint _index) external {
        require(_index < candidates.length, "Invalid candidate index");

        candidates[_index].voteCount++;
    }

}
