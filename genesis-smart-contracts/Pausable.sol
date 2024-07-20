// genesis-smart-contracts/Pausable.sol
pragma solidity ^0.8.0;

contract Pausable {
    bool public paused;

    modifier whenNotPaused() {
        require(!paused, "Contract is paused");
        _;
    }

    modifier whenPaused() {
        require(paused, "Contract is not paused");
        _;
    }

    function pause() public onlyOwner {
        paused = true;
        emit Pause();
    }

    function unpause() public onlyOwner {
        paused = false;
        emit Unpause();
    }

    event Pause();
    event Unpause();
}
