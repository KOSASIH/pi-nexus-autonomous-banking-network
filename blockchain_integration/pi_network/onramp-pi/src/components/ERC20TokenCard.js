// onramp-pi/src/components/ERC20TokenCard.js

import React from 'react';
import { Card, CardBody, CardHeader, CardFooter } from 'reactstrap';
import { useWeb3React } from '@web3-react/core';
import { Web3Utils } from '../utils/web3';
import { config } from '../config';

const ERC20TokenCard = () => {
  const { account } = useWeb3React();
  const [balance, setBalance] = useState(0);

  useEffect(() => {
    const fetchBalance = async () => {
      const balance = await Web3Utils.getERC20Balance(config.erc20TokenAddress, account);
      setBalance(balance);
    };
    fetchBalance();
  }, [account]);

  return (
    <Card>
      <CardHeader>
        <h5>{config.erc20TokenSymbol} Token</h5>
      </CardHeader>
      <CardBody>
        <p>Balance: {balance} {config.erc20TokenSymbol}</p>
      </CardBody>
      <CardFooter>
        <button onClick={() => console.log('Buy token')}>Buy</button>
        <button onClick={() => console.log('Sell token')}>Sell</button>
      </CardFooter>
    </Card>
  );
};

export default ERC20TokenCard;
