// src/currency/currencyService.ts
import axios from 'axios';

const getExchangeRate = async (from: string, to: string) => {
    const response = await axios.get(`https://api.exchangerate-api.com/v4/latest/${from}`);
    return response.data.rates[to];
};

export const convertCurrency = async (amount: number, from: string, to: string) => {
    const rate = await getExchangeRate(from, to);
    return amount * rate;
};
