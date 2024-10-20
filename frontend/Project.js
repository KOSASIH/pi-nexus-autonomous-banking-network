import React, { useState, useEffect } from 'react';

const Project = ({ project }) => {
  const [invested, setInvested] = useState(false);

  useEffect(() => {
    // Check if the user has already invested in the project
    impactInvestingContract.methods.getInvestments(web3.eth.accounts[0]).call().then((investments) => {
      for (let i = 0; i < investments.length; i++) {
        if (investments[i].projectId === project.id) {
          setInvested(true);
          break;
        }
      }
    });
  }, [project]);

  const handleInvest = async () => {
    await invest(project.id, 1);
    setInvested(true);
  };

  return (
    <div>
      <h2>{project.name}</h2>
      <p>{project.description}</p>
      <p>Goal: {project.goal} ETH</p>
      <p>Raised: {project.raised} ETH</p>
      {invested ? (
        <p>You have already invested in this project.</p>
      ) : (
        <button onClick={handleInvest}>Invest 1 ETH</button>
      )}
    </div>
  );
};

export default Project;
