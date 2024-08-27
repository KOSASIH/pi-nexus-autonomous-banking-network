import React, { useState, useEffect } from 'react';
import { getPiNexusAccountId, getPiNexusAccountBalance, transferPiNexusFunds } from '../api/pi-nexus-api';
import { getTraditionalFinanceAccountId, getTraditionalFinanceAccountBalance, transferTraditionalFinanceFunds } from '../api/traditional-finance-api';

interface Props {
  piNexusAccountId: string;
  traditionalFinanceAccountId: string;
}

const PiNexusTraditionalFinanceBridge: React.FC<Props> = ({ piNexusAccountId, traditionalFinanceAccountId }) => {
  const [piNexusAccountBalance, setPiNexusAccountBalance] = useState<number>(0);
  const [traditionalFinanceAccountBalance, setTraditionalFinanceAccountBalance] = useState<number>(0);
  const [transferAmount, setTransferAmount] = useState<number>(0);

  useEffect(() => {
    getPiNexusAccountBalance(piNexusAccountId).then((response) => {
      setPiNexusAccountBalance(response.data.balance);
    });
  }, [piNexusAccountId]);

  useEffect(() => {
    getTraditionalFinanceAccountBalance(traditionalFinanceAccountId).then((response) => {
      setTraditionalFinanceAccountBalance(response.data.balance);
    });
  }, [traditionalFinanceAccountId]);

  const handleTransferFunds = () => {
    transferPiNexusFunds(piNexusAccountId, traditionalFinanceAccountId, transferAmount).then((response) => {
      console.log(response.data);
    });
    transferTraditionalFinanceFunds(traditionalFinanceAccountId, piNexusAccountId, transferAmount).then((response) => {
      console.log(response.data);
    });
  };

  return (
    <div>
      <h2>Pi Nexus Traditional Finance Bridge</h2>
      <p>Pi Nexus Account Balance: {piNexusAccountBalance}</p>
      <p>Traditional Finance Account Balance: {traditionalFinanceAccountBalance}</p>
      <input
        type="number"
        value={transferAmount}
        onChange={(e) => setTransferAmount(e.target.valueAsNumber)}
        placeholder="Transfer Amount"
      />
      <button onClick={handleTransferFunds}>Transfer Funds</button>
    </div>
  );
};

export default PiNexusTraditionalFinanceBridge;
