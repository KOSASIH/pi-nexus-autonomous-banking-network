import React from 'react';
import TraditionalFinanceIntegration from '../components/TraditionalFinanceIntegration';

interface Props {
  piNexusAccountId: string;
  traditionalFinanceAccountId: string;
}

const TraditionalFinanceContainer: React.FC<Props> = ({ piNexusAccountId, traditionalFinanceAccountId }) => {
  return (
    <div>
      <TraditionalFinanceIntegration piNexusAccountId={piNexusAccountId} traditionalFinanceAccountId={traditionalFinanceAccountId} />
    </div>
  );
};

export default TraditionalFinanceContainer;
