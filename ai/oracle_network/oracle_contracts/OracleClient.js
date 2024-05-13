const OracleAggregator = require('./oracle-aggregator/OracleAggregator');
const Web3 = require('web3');

const web3 = new Web3('HTTP://YOUR_RPC_URL');
const oracleAggregatorAddress = 'YOUR_ORACLE_AGGREGATOR_ADDRESS';
const oracleAggregator = new OracleAggregator();

async function updateAggregatedData(api) {
    const oracleAggregatorContract = new web3.eth.Contract(OracleAggregator.abi, oracleAggregatorAddress);
    await oracleAggregator.updateAggregatedData(oracleAggregatorContract, api);
}

updateAggregatedData('weather');
updateAggregatedData('stock-market');
