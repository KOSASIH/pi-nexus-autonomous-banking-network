import { LiquidityModel } from '../models/LiquidityModel';
import { ExchangeModel } from '../models/ExchangeModel';
import exchanges from '../data/exchanges.json';
import markets from '../data/markets.json';

const liquidityModel = new LiquidityModel();
const exchangeModels = exchanges.map(exchange => new ExchangeModel(exchange));

async function aggregateLiquidity() {
  const liquidityData = [];
  for (const market of markets) {
    const exchangeModel = exchangeModels.find(exchangeModel => exchangeModel.exchange.name === market.exchange);
    const ticker = await exchangeModel.getTicker(market.symbol);
    const liquidity = liquidityModel.predict([ticker.bid, ticker.ask, ...]);
    liquidityData.push({ market: market.symbol, liquidity });
  }
  return liquidityData;
}

async function executeTrade(market, side, quantity, price) {
  const exchangeModel = exchangeModels.find(exchangeModel => exchangeModel.exchange.name === market.exchange);
  await exchangeModel.placeOrder(market.symbol, side, quantity, price);
}

async function main() {
  const liquidityData = await aggregateLiquidity();
  for (const { market, liquidity } of liquidityData) {
    if (liquidity > 0.5) {
      await executeTrade(market, 'buy', 0.1, 10000);
    } else {
      await executeTrade(market, 'sell', 0.1, 10000);
    }
  }
}

main();
