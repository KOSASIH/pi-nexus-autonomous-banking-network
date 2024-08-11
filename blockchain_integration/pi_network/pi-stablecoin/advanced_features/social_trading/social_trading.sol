pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract SocialTrading {
    // Mapping of trader profiles
    mapping (address => TraderProfile) public traderProfiles;

    // Function to follow a trader
    function followTrader(address trader) public {
        // Update follower count for trader
        TraderProfile storage traderProfile = traderProfiles[trader];
        traderProfile.followers++;
    }
}
