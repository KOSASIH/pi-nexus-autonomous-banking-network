pragma solidity ^0.8.0;

contract Treasury {
    // Mapping of asset symbols to their corresponding treasury balances
    mapping (string => uint256) public treasuryBalances;

    // Event emitted when a deposit is made to the treasury
    event DepositMade(string assetSymbol, uint256 amount);

    // Event emitted when a withdrawal is made from the treasury
    event WithdrawalMade(string assetSymbol, uint256 amount);

    // Function to deposit funds into the treasury
    function deposit(string memory assetSymbol, uint256 amount) public {
        // Update the treasury balance
        treasuryBalances[assetSymbol] += amount;

        // Emit the DepositMade event
        emit DepositMade(assetSymbol, amount);
    }

    // Function to withdraw funds from the treasury
    function withdraw(string memory assetSymbol, uint256 amount) public {
        // Check if the treasury has sufficient balance
        require(treasuryBalances[assetSymbol] >= amount, "Insufficient balance");

        // Update the treasury balance
        treasuryBalances[assetSymbol] -= amount;

        // Emit the WithdrawalMade event
        emit WithdrawalMade(assetSymbol, amount);
    }

    // Function to get the treasury balance for a specific asset
    function getTreasuryBalance(string memory assetSymbol) public view returns (uint256) {
        return treasuryBalances[assetSymbol];
    }
}
