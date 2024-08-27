import React, { useState, useEffect } from 'react';
import { getBankAccountInfo, getStockExchangeQuotes, executeTrade, getAccountBalance, transferFunds } from '../api/traditional-finance-api';

interface Props {
  piNexusAccountId: string;
  traditionalFinanceAccountId: string;
}

const TraditionalFinanceIntegration: React.FC<Props> = ({ piNexusAccountId, traditionalFinanceAccountId }) => {
  const [bankAccountInfo, setBankAccountInfo] = useState<any>(null);
  const [stockExchangeQuotes, setStockExchangeQuotes] = useState<any>(null);
  const [accountBalance, setAccountBalance] = useState<number>(0);
  const [transferAmount, setTransferAmount] = useState<number>(0);

  useEffect(() => {
    getBankAccountInfo(traditionalFinanceAccountId).then((response) => {
      setBankAccountInfo(response.data);
    });
  }, [traditionalFinanceAccountId]);

  useEffect(() => {
    getStockExchangeQuotes('AAPL').then((response) => {
      setStockExchangeQuotes(response.data);
    });
  }, []);

  const handleExecuteTrade = () => {
    executeTrade('AAPL', 10, 150).then((response) => {
      console.log(response.data);
    });
  };

  const handleGetAccountBalance = () => {
    getAccountBalance(traditionalFinanceAccountId).then((response) => {
      setAccountBalance(response.data.balance);
    });
  };

  const handleTransferFunds = () => {
    transferFunds(traditionalFinanceAccountId, piNexusAccountId, transferAmount).then((response) => {
      console.log(response.data);
    });
  };

  return (
    <div>
      <h2>Traditional Finance Integration</h2>
      <p>Bank Account Info:</p>
      <ul>
        {bankAccountInfo && (
          <li>
            Account Number: {bankAccountInfo.accountNumber}
            <br />
            Account Type: {bankAccountInfo.accountType}
          </li>
        )}
      </ul>
      <p>Stock Exchange Quotes:</p>
      <ul>
        {stockExchangeQuotes && (
          <li>
            Symbol: {stockExchangeQuotes.symbol}
            <br />
            Price: {stockExchangeQuotes.price}
          </li>
        )}
      </ul>
      <p>Account Balance: {accountBalance}</p>
      <input
        type="number"
        value={transferAmount}
        onChange={(e) => setTransferAmount(e.target.valueAsNumber)}
        placeholder="Transfer Amount"
      />
      <button onClick={handleTransferFunds}>Transfer Funds</button>
      <button onClick={handleExecuteTrade}>Execute Trade</button>
      <button onClick={handleGetAccountBalance}>Get Account Balance</button>
    </div>
  );
};

export default TraditionalFinanceIntegration;
