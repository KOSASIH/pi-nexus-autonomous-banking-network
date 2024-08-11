import axios from 'axios';
import { config } from '../config';

const binanceSmartChainApiUrl = config.dex.binanceSmartChain.apiUrl;
const binanceSmartChainApiKey = config.dex.binanceSmartChain.apiKey;

export async function getBinanceSmartChainBalance(address) {
  const response = await axios.get(`${binanceSmartChainApiUrl}/account/${address}/balance`, {
    headers: {
      'X-API-KEY': binanceSmartChainApiKey
    }
  });
  return response.data.balance;
}
