import axios from 'axios';

const piNexusApiEndpoint = 'https://api.pi-nexus.io';

export const getPiNexusAccountId = () => {
  return axios.get(`${piNexusApiEndpoint}/accounts/me`);
};

export const getPiNexusAccountBalance = (accountId: string) => {
  return axios.get(`${piNexusApiEndpoint}/accounts/${accountId}/balance`);
};

export const transferPiNexusFunds = (fromAccountId: string, toAccountId: string, amount: number) => {
  return axios.post(`${piNexusApiEndpoint}/accounts/${fromAccountId}/transfers`, {
    toAccountId,
    amount,
  });
};
