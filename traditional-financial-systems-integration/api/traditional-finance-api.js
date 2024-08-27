import axios from 'axios';

const traditionalFinanceApiEndpoint = 'https://api.traditional-finance.io';

export const getBankAccountInfo = (accountId: string) => {
  return axios.get(`${traditionalFinanceApiEndpoint}/bank/accounts/${accountId}`);
};

export const getStockExchangeQuotes = (symbol: string) => {
  return axios.get(`${traditionalFinanceApiEndpoint}/stock-exchange/quotes/${symbol}`);
};

export const executeTrade = (symbol: string, quantity: number, price: number) => {
  return axios.post(`${traditionalFinanceApiEndpoint}/stock-exchange/trades`, {
    symbol,
    quantity,
    price,
  });
};

export const getAccountBalance = (accountId: string) => {
  return axios.get(`${traditionalFinanceApiEndpoint}/bank/accounts/${accountId}/balance`);
};

export const transferFunds = (fromAccountId: string, toAccountId: string, amount: number) => {
  return axios.post(`${traditionalFinanceApiEndpoint}/bank/transfers`, {
    fromAccountId,
    toAccountId,
    amount,
  });
};
