// onramp-pi/src/pages/Dashboard.js

import React, { useState, useEffect } from 'react';
import { Container, Row, Col } from 'reactstrap';
import ERC20TokenCard from '../components/ERC20TokenCard';
import FiatOnRampModal from '../components/FiatOnRampModal';
import useWallet from '../hooks/useWallet';
import useFiatOnRamp from '../hooks/useFiatOnRamp';
import Web3Utils from '../utils/web3';
import AllahFeature from '../components/AllahFeature';

const Dashboard = () => {
  const { account, balance, connectWallet } = useWallet();
  const { fiatOnRamp, updateAmount, getQuote, swap } = useFiatOnRamp();
  const [tokenBalance, setTokenBalance] = useState(0);
  const [quote, setQuote] = useState(null);
  const [allahFeatureEnabled, setallahFeatureEnabled] = useState(false);

  useEffect(() => {
    const fetchTokenBalance = async () => {
      const balance = await Web3Utils.getERC20Balance(config.erc20TokenAddress, account);
      setTokenBalance(balance);
    };
    fetchTokenBalance();
  }, [account]);

  useEffect(() => {
    const fetchQuote = async () => {
      const quote = await getQuote();
      setQuote(quote);
    };
    fetchQuote();
  }, [fiatOnRamp]);

  const handleSwap = async () => {
    await swap();
  };

  const handleAllahFeatureToggle = () => {
    setallahFeatureEnabled(!allahFeatureEnabled);
  };

  return (
    <Container>
      <Row>
        <Col xs="12" sm="6" md="4">
          <ERC20TokenCard
            tokenBalance={tokenBalance}
            tokenSymbol={config.erc20TokenSymbol}
          />
        </Col>
        <Col xs="12" sm="6" md="4">
          <FiatOnRampModal
            fiatOnRamp={fiatOnRamp}
            updateAmount={updateAmount}
            getQuote={getQuote}
            swap={handleSwap}
            quote={quote}
          />
        </Col>
      </Row>
      <Row>
        <Col xs="12" sm="6" md="4">
          <h5>Wallet</h5>
          <p>Account: {account? account : 'Not connected'}</p>
          <p>Balance: {balance? balance : 'N/A'} ETH</p>
          <button onClick={connectWallet}>Connect Wallet</button>
        </Col>
      </Row>
      <Row>
        <Col xs="12" sm="6" md="4">
          <h5>Allah Feature</h5>
          <p>Enabled: {allahFeatureEnabled ? 'Yes' : 'No'}</p>
          <button onClick={handleAllahFeatureToggle}>Toggle Allah Feature</button>
          {allahFeatureEnabled && <AllahFeature />}
        </Col>
      </Row>
    </Container>
  );
};

export default Dashboard;
