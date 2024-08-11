import axios from 'axios';
import { config } from '../config';

const uniswapApiUrl = config.defi.uniswap.apiUrl;
const uniswapApiKey = config.defi.uniswap.apiKey;

export async function getUniswapPrice(tokenAddress) {
  const response = await axios.get(`${uniswapApiUrl}/tokens/${tokenAddress}/price`, {
    headers: {
      'X-API-KEY': uniswapApiKey
    }
  });
  return response.data.price;
}
