pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract DecentralizedPredictionMarket {
    // Mapping of user addresses to balances
    mapping (address => uint256) public balances;

    // Mapping of outcome choices to outcome probabilities
    mapping (uint256 => uint256) public outcomeProbabilities;

    // Event emitted when a new bet is placed
    event BetPlaced(address user, uint256 outcome, uint256 amount);

    // Function to place a new bet
    function placeBet(uint256 _outcome, uint256 _amount) public {
        // Check if bet amount is valid
        require(_amount > 0, "Invalid bet amount");

        // Check if user has enough balance
        require(balances[msg.sender] >= _amount, "Insufficient balance");

        // Update outcome probability
        outcomeProbabilities[_outcome] = outcomeProbabilities[_outcome].add(_amount);

        // Update user balance
        balances[msg.sender] = balances[msg.sender].sub(_amount);

        // Emit bet placed event
        emit BetPlaced(msg.sender, _outcome, _amount);
    }

    // Function to settle bets
    function settleBets() public {
        // Calculate total bet amount
        uint256 totalBetAmount = 0;

        // Iterate through all outcomes and calculate total bet amount
        for (uint256 outcome : allOutcomes) {
            totalBetAmount = totalBetAmount.add(outcomeProbabilities[outcome]);
        }

        // Distribute funds to winning outcomes
        for (uint256 outcome : allOutcomes) {
            uint256 outcomeAmount = outcomeProbabilities[outcome].mul(100).div(totalBetAmount);

            // Transfer funds to winning outcome
            msg.sender.transfer(outcomeAmount);
        }
    }
}
