import React, { useState } from 'react';
import { useDispatch } from 'react-redux';
import { createContract } from '../actions/contractActions';

const CreateContract = () => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const dispatch = useDispatch();

  const handleSubmit = (event) => {
    event.preventDefault();
    dispatch(createContract({ name, description }));
  };

  return (
    <div>
      <h2>Create Contract</h2>
      <form onSubmit={handleSubmit}>
        <label>
          Name:
          <input type="text" value={name} onChange={(event) => setName(event.target.value)} />
        </label>
        <label>
          Description:
          <textarea value={description} onChange={(event) => setDescription(event.target.value)} />
        </label>
        <button type="submit">Create Contract</button>
      </form>
    </div>
  );
};

export default CreateContract;
