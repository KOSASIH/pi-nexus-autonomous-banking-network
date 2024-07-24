import React, { useState, useEffect } from 'react';
import { useWeb3React } from '@web3-react/core';
import { AstralPlaneGovernance } from '../contracts/AstralPlaneGovernance';

const AstralPlaneGovernance = () => {
  const { account, library } = useWeb3React();
  const [proposals, setProposals] = useState([]);
  const [vote, setVote] = useState({});

  useEffect(() => {
    const fetchProposals = async () => {
      const proposals = await AstralPlaneGovernance.getProposals();
      setProposals(proposals);
    };
    fetchProposals();
  }, [AstralPlaneGovernance]);

  const handleVote = async (proposalId, support) => {
    await AstralPlaneGovernance.vote(proposalId, support);
    setVote({ [proposalId]: support });
  };

  const handleExecute = async (proposalId) => {
    await AstralPlaneGovernance.execute(proposalId);
  };

  return (
    <div>
      <h1>AstralPlane Governance</h1>
      <ul>
        {proposals.map((proposal, index) => (
          <li key={index}>
            <p>Proposal {index + 1}: {proposal.description}</p>
            <p>Voting Period: {proposal.votingPeriod}</p>
            <p>Execution Delay: {proposal.executionDelay}</p>
            <button onClick={() => handleVote(proposal.id, true)}>Vote For</button>
            <button onClick={() => handleVote(proposal.id, false)}>Vote Against</button>
            {vote[proposal.id] && <p>You voted {vote[proposal.id] ? 'for' : 'against'} this proposal</p>}
            {proposal.executionDelay <= block.timestamp && <button onClick={() => handleExecute(proposal.id)}>Execute</button>}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default AstralPlaneGovernance;
