pragma solidity ^0.8.0;

contract EnergyTradingContract {
    // Mapping of energy producers to their energy production
    mapping (address => uint256) public energyProducers;

    // Mapping of energy consumers to their energy consumption
    mapping (address => uint256) public energyConsumers;

    // Event emitted when energy is traded
    event EnergyTraded(address producer, address consumer, uint256 amount);

    // Function to register energy producer
    function registerProducer(address producer) public {
        energyProducers[producer] = 0;
    }

    // Function to register energy consumer
    function registerConsumer(address consumer) public {
        energyConsumers[consumer] = 0;
    }

    // Function to trade energy
    function tradeEnergy(address producer, address consumer, uint256 amount) public {
        require(energyProducers[producer] >= amount, "Producer does not have enough energy");
        require(energyConsumers[consumer] >= amount, "Consumer does not have enough energy");

        energyProducers[producer] -= amount;
        energyConsumers[consumer] += amount;

        emit EnergyTraded(producer, consumer, amount);
    }
}
