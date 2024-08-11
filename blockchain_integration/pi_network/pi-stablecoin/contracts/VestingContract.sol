pragma solidity ^0.8.0;

contract VestingContract {
    // Mapping of user addresses to their vesting schedules
    mapping (address => VestingSchedule) public vestingSchedules;

    // Event emitted when a user's vesting schedule is updated
    event VestingScheduleUpdated(address user, uint256 vestingAmount, uint256 vestingPeriod);

    // Event emitted when a user's vesting period is completed
    event VestingPeriodCompleted(address user, uint256 vestingAmount);

    // Struct to represent a vesting schedule
    struct VestingSchedule {
        uint256 vestingAmount;
        uint256 vestingPeriod;
        uint256 startTime;
    }

    // Function to create a new vesting schedule for a user
    function createVestingSchedule(address user, uint256 vestingAmount, uint256 vestingPeriod) public {
        // Check if the user already has a vesting schedule
        require(vestingSchedules[user].vestingAmount == 0, "User already has a vesting schedule");

        // Create a new vesting schedule
        vestingSchedules[user] = VestingSchedule(vestingAmount, vestingPeriod, block.timestamp);

        // Emit the VestingScheduleUpdated event
        emit VestingScheduleUpdated(user, vestingAmount, vestingPeriod);
    }

    // Function to vest tokens for a user
    function vest(address user) public {
        // Check if the user has a vesting schedule
        require(vestingSchedules[user].vestingAmount > 0, "User does not have a vesting schedule");

        // Check if the vesting period has completed
        require(block.timestamp >= vestingSchedules[user].startTime + vestingSchedules[user].vestingPeriod, "Vesting period has not completed");

        // Vest the tokens
        vestingSchedules[user].vestingAmount = 0;

        // Emit the VestingPeriodCompleted event
        emit VestingPeriodCompleted(user, vestingSchedules[user].vestingAmount);
    }

    // Function to get a user's vesting schedule
    function getVestingSchedule(address user) public view returns (VestingSchedule memory) {
        return vestingSchedules[user];
    }
}
