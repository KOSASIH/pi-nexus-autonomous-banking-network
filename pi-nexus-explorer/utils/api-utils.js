import axios from 'axios';

const apiEndpoint = 'https://api.pi-nexus.io';

export const getBlock = (blockNumber: string) => {
  return axios.get(`${apiEndpoint}/blocks/${blockNumber}`);
};

export const getTransaction = (transactionHash: string) => {
  return axios.get(`${apiEndpoint}/transactions/${transactionHash}`);
};

export const getContract = (contractAddress: string) => {
  return axios.get(`${apiEndpoint}/contracts/${contractAddress}`);
};

export const getBlocks = () => {
  return axios.get(`${apiEndpoint}/blocks`);
};

export const getTransactions = () => {
  return axios.get(`${apiEndpoint}/transactions`);
};

export const getContracts = () => {
  return axios.get(`${apiEndpoint}/contracts`);
};
