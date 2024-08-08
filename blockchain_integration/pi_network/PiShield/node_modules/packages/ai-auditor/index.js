const { NeuralNetwork } = require('./models/neural_network');
const { RuleBased } = require('./models/rule_based');

class AIAuditor {
    async auditContract(contractAddress, data) {
        const neuralNetwork = new NeuralNetwork();
        const ruleBased = new RuleBased();

        const neuralNetworkResult = await neuralNetwork.audit(contractAddress, data);
        const ruleBasedResult = await ruleBased.audit(contractAddress, data);

        return { neuralNetworkResult, ruleBasedResult };
    }
}

module.exports = AIAuditor;
