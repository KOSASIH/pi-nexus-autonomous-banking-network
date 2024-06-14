import { Trade, TradeType } from '@uniswap/sdk';
import { TokenAmount, Pair, Route, Trade as UniswapTrade } from '@uniswap/v2-sdk';
import { Token, TokenAmount as TokenAmountV2 } from '@uniswap/sdk-core';
import { BigNumber } from 'bignumber.js';
import { ethers } from 'ethers';
import { ZERO_ADDRESS } from '@uniswap/v2-core';
import { ChainId } from '@uniswap/sdk-core';
import { useContractFunction, useEthers, useCall } from '@usedapp/core';
import { Contract } from '@ethersproject/contracts';
import { abi as IUniswapV2Router02ABI } from '@uniswap/v2-periphery/build/IUniswapV2Router02.json';
import { abi as IUniswapV2PairABI } from '@uniswap/v2-core/build/IUniswapV2Pair.json';

const IUniswapV2Router02Address = '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D';

const IUniswapV2Router02Interface = new ethers.utils.Interface(IUniswapV2Router02ABI);
const IUniswapV2PairInterface = new ethers.utils.Interface(IUniswapV2PairABI);

const routerContract = new Contract(
  IUniswapV2Router02Address,
  IUniswapV2Router02Interface,
  ethers.getDefaultProvider('rinkeby')
);

function useTrade(tokenIn: Token, tokenOut: Token, amountIn: TokenAmount, tradeType: TradeType) {
  const { chainId } = useEthers();

  const [route, setRoute] = useState<Route | null>(null);
  const [trade, setTrade] = useState<Trade | null>(null);

  const { value: pairAddress } = useCall(
    {
      contract: routerContract,
      method: 'getPair',
      args: [tokenIn.address, tokenOut.address]
    },
    {
      chainId
    }
  );

  useEffect(() => {
    if (pairAddress) {
      const pairContract = new Contract(pairAddress, IUniswapV2PairInterface, ethers.getDefaultProvider('rinkeby'));
      const { value: reserves } = useCall(
        {
          contract: pairContract,
          method: 'getReserves'
        },
        {
          chainId
        }
      );

      const tokenAmountIn = new TokenAmount(tokenIn, amountIn.raw.toString());
      const tokenAmountOut = Trade.getAmountOut(tokenAmountIn, reserves.reserve0, reserves.reserve1);

      const route = new Route([new Pair(tokenIn, tokenOut, reserves.reserve0, reserves.reserve1)], tokenIn, tokenOut);
      const trade = new Trade(route, tokenAmountIn, tokenAmountOut, tradeType);

      setRoute(route);
      setTrade(trade);
    }
  }, [pairAddress, tokenIn, tokenOut, amountIn, tradeType]);

  return { route, trade };
}

export default useTrade;
