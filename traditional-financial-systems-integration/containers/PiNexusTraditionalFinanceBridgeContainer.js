import React from 'react';
import PiNexusTraditionalFinanceBridge from '../components/PiNexusTraditionalFinanceBridge';

interface Props {
  piNexusAccountId: string;
  traditionalFinanceAccountId: string;
}

const PiNexusTraditionalFinanceBridgeContainer: React.FC<Props> = ({ piNexusAccountId, traditionalFinanceAccountId }) => {
  return (
    <div>
      <PiNexusTraditionalFinanceBridge piNexusAccountId={piNexusAccountId} traditionalFinanceAccountId={traditionalFinanceAccountId} />
    </div>
  );
};

export default PiNexusTraditionalFinanceBridgeContainer;
