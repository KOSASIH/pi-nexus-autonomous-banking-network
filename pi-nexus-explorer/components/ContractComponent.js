import React from 'react';
import { Contract } from '../types';

interface Props {
  contract: Contract;
}

const ContractComponent: React.FC<Props> = ({ contract }) => {
  return (
    <div>
      <h2>Contract {contract.address}</h2>
      <p>Bytecode: {contract.bytecode}</p>
      <p>ABI: {contract.abi}</p>
    </div>
  );
};

export default ContractComponent;
