// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Crowdfunding {
    struct Campaign {
        address creator;
        uint256 goal;
        uint256 raisedAmount;
        uint256 deadline;
        bool isCompleted;
    }

    mapping(uint256 => Campaign) public campaigns;
    uint256 public campaignCount;

    function createCampaign(uint256 goal, uint256 duration) external {
        require(goal > 0, "Goal must be greater than 0");
        campaignCount++;
        campaigns[campaignCount] = Campaign(msg.sender, goal, 0, block.timestamp + duration, false);
    }

    function contribute(uint256 campaignId) external payable {
        Campaign storage campaign = campaigns[campaignId];
        require(block.timestamp < campaign.deadline, "Campaign has ended");
        require(!campaign.isCompleted, "Campaign already completed");

        campaign.raisedAmount += msg.value;
    }

    function finalizeCampaign(uint256 campaignId) external {
        Campaign storage campaign = campaigns[campaignId];
        require(block.timestamp >= campaign.deadline, "Campaign is still ongoing");
        require(!campaign.isCompleted, "Campaign already completed");

        campaign.isCompleted = true;
        if (campaign.raisedAmount >= campaign.goal) {
            payable(campaign.creator).transfer(campaign.raisedAmount);
        }
    }
}
