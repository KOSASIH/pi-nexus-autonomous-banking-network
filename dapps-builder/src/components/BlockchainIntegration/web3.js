import Web3 from 'web3';

const web3 = new Web3(Web3.givenProvider || 'http://localhost:8545');

export const getAccounts = async () => {
    return await web3.eth.getAccounts();
};

export const getNetworkId = async () => {
    return await web3.eth.net.getId();
};

export default web3;
