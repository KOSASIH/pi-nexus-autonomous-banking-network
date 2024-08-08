import React from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { fetchContract } from '../actions/contractActions';

const ContractDetails = ({ match }) => {
  const contract = useSelector((state) => state.contract);
  const dispatch = useDispatch();

  useEffect(() => {
    dispatch(fetchContract(match.params.id));
  }, []);

  return (
    <div>
      <h2>Contract Details</h2>
      <p>{contract.name}</p>
      <p>{contract.description}</p>
    </div>
  );
};

export default ContractDetails;
