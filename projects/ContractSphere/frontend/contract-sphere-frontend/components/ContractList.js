import React, { useState, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { fetchContracts } from '../actions/contractActions';

const ContractList = () => {
  const contracts = useSelector((state) => state.contracts);
  const dispatch = useDispatch();

  useEffect(() => {
    dispatch(fetchContracts());
  }, []);

  return (
    <div>
      <h2>Contract List</h2>
      <ul>
        {contracts.map((contract) => (
          <li key={contract.id}>
            <Link to={`/contracts/${contract.id}`}>{contract.name}</Link>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default ContractList;
