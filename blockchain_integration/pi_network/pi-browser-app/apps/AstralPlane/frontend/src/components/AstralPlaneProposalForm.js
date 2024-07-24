import React, { useState } from 'react';

const AstralPlaneProposalForm = () => {
  const [targets, setTargets] = useState([]);
  const [values, setValues] = useState([]);
  const [calldatas, setCalldatas] = useState([]);
  const [description, setDescription] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    const proposalId = await fetch('/api/astralplane-governance/proposal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ targets, values, calldatas, description }),
    });
    console.log(`Proposal created with ID ${proposalId}`);
  };

  return (
    <form onSubmit={handleSubmit}>
      <label>Targets:</label>
      <input type="text" value={targets} onChange={(event) => setTargets(event.target.value.split(','))} />
      <br />
      <label>Values:</label>
      <input type="text" value={values} onChange={(event) => setValues(event.target.value.split(','))} />
      <br />
      <label>Calldatas:</label>
      <input type="text" value={calldatas} onChange={(event) => setCalldatas(event.target.value.split(','))} />
      <br />
      <label>Description:</label>
      <textarea value={description} onChange={(event) => setDescription(event.target.value)} />
      <br />
      <button type="submit">Create Proposal</button>
    </form>
  );
};

export default AstralPlaneProposalForm;
