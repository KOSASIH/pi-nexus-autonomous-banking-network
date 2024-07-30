pragma solidity ^0.8.0;

contract GamificationContract {
    mapping (address => uint256) public scores;

    function updateScore(uint256 _score) public {
        scores[msg.sender] += _score;
    }

    function getScore() public view returns (uint256) {
        return scores[msg.sender];
    }
}
