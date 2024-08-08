const { Executor } = require('./executor');

class ContractExecutor {
    async executeContract(contractAddress, data) {
        const executor = new Executor();
        return executor.execute(contractAddress, data);
    }
}

module.exports = ContractExecutor;
