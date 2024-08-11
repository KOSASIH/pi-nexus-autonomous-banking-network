import { getUniswapPrice } from './protokol/defi/uniswap';
import { getBinanceSmartChainBalance } from './protokol/dex/binance-smart-chain';
import { getChainlinkPriceFeed } from './protokol/oracle/chainlink';
import { getCurveAPY } from './protokol/defi/curve';
import { getHuobiBalance } from './protokol/dex/huobi';
import { getAaveLendingRate } from './protokol/oracle/aave';

async function main() {
  const piStablecoinAddress = '0x...';
  const tokenAddress = '0x...';

  const uniswapPrice = await getUniswapPrice(tokenAddress);
  console.log(`Uniswap price: ${uniswapPrice}`);

  const binanceSmartChainBalance = await getBinanceSmartChainBalance(piStablecoinAddress);
  console.log(`Binance Smart Chain balance: ${binanceSmartChainBalance}`);

  const chainlinkPriceFeed = await getChainlinkPriceFeed(tokenAddress);
  console.log(`Chainlink price feed: ${chainlinkPriceFeed}`);

  const curveAPY = await getCurveAPY(tokenAddress);
  console.log(`Curve APY: ${curveAPY}`);

  const huobiBalance = await getHuobiBalance(piStablecoinAddress);
  console.log(`Huobi balance: ${huobiBalance}`);

  const aaveLendingRate = await getAaveLendingRate(tokenAddress);
  console.log(`Aave lending rate: ${aaveLendingRate}`);
}

main();
