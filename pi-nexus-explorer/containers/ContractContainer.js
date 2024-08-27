import React, { useState, useEffect } from 'react';
import { Contract } from '../types';
import { apiUtils } from '../utils';
import ContractComponent from '../components/ContractComponent';

interface Props {
  match: {
    params: {
      contractAddress: string;
    };
  };
}

const ContractContainer: React.FC<Props> = ({ match }) => {
  const [contract, setContract] = useState<Contract | null>(null);

  useEffect(() => {
    apiUtils.getContract(match.params.contractAddress).then((contract) => setContract(contract));
  }, [match.params.contractAddress]);

  return (
    <div>
      {contract ? <ContractComponent contract={contract} /> : <p>Loading...</p>}
    </div>
  );
};

export default ContractContainer;
