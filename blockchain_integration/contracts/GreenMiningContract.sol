pragma solidity ^0.8.0;

contract GreenMiningContract {
    // Mapping of miner addresses to their renewable energy sources
    mapping(address => RenewableEnergySource) public energySources;

    // Mapping of miner addresses to their bonus balances
    mapping(address => uint256) public bonusBalances;

    // Event emitted when a miner registers a new energy source
    event EnergySourceRegistered(address miner, string energySourceName);

    // Event emitted when a miner claims their bonus
    event BonusClaimed(address miner, uint256 amount);

    // Struct to represent a renewable energy source
    struct RenewableEnergySource {
        string name;
        uint256 energyOutput;
    }

    // Function to register a new energy source
    function registerEnergySource(string memory _name, uint256 _energyOutput) public {
        RenewableEnergySource memory newSource = RenewableEnergySource(_name, _energyOutput);
        energySources[msg.sender] = newSource;
        emit EnergySourceRegistered(msg.sender, _name);
    }

    // Function to calculate a miner's bonus
    function calculateBonus() public view returns (uint256) {
        RenewableEnergySource storage energySource = energySources[msg.sender];
        return energySource.energyOutput * 10; // 10 Pi coins per unit of energy output
    }

    // Function to claim a miner's bonus
    function claimBonus() public {
        uint256 bonus = calculateBonus();
        bonusBalances[msg.sender] += bonus;
        emit BonusClaimed(msg.sender, bonus);
    }

    // Function to view a miner's bonus balance
    function viewBonusBalance() public view returns (uint256) {
        return bonusBalances[msg.sender];
    }
}
