import axios from 'axios';
import { config } from '../config';

const chainlinkApiUrl = config.oracle.chainlink.apiUrl;
const chainlinkApiKey = config.oracle.chainlink.apiKey;

export async function getChainlinkPriceFeed(tokenAddress) {
  const response = await axios.get(`${chainlinkApiUrl}/price-feeds/${tokenAddress}`, {
    headers: {
      'X-API-KEY': chainlinkApiKey
    }
  });
  return response.data.priceFeed;
}
