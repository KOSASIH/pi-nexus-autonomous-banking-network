pragma solidity ^0.8.0;

contract Crowdfunding {

    // The funding goal
    uint public fundingGoal;

    // The amount raised
    uint public amountRaised;

    // The deadline
    uint public deadline;

    // The status
    bool public status;

    // The function to initialize the contract
    constructor(uint _fundingGoal, uint _deadline) {
        fundingGoal = _fundingGoal;
        deadline = _deadline;
        status = false;
    }

    // The function to contribute
    function contribute() external payable {
        require(status == false, "Crowdfunding already finished");
        require(block.timestamp <= deadline, "Deadline already passed");

        amountRaised += msg.value;

        if (amountRaised >= fundingGoal) {
            status = true;
        }
    }

    // The function to refund contributions
    function refund() external {
        require(status == true, "Crowdfunding not finished yet");

        msg.sender.transfer(msg.value);
    }

}
