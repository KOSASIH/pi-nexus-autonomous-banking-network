import express from 'express';
import Web3 from 'web3';
import { AstralPlaneGovernance } from '../contracts/AstralPlaneGovernance';

const app = express();
const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/454c372bb595486f90fc6295b128695c'));

app.get('/api/astralplane-governance/proposals', async (req, res) => {
  const proposals = await AstralPlaneGovernance.getProposals();
  res.json({ proposals });
});

app.post('/api/astralplane-governance/proposal', async (req, res) => {
  const { targets, values, calldatas, description } = req.body;
  const proposalId = await AstralPlaneGovernance.propose(targets, values, calldatas, description);
  res.json({ proposalId });
});

app.get('/api/astralplane-governance/proposal/:id', async (req, res) => {
  const proposalId = req.params.id;
  const proposal = await AstralPlaneGovernance.getProposal(proposalId);
  res.json({ proposal });
});

app.post('/api/astralplane-governance/vote', async (req, res) => {
  const { proposalId, support } = req.body;
  await AstralPlaneGovernance.vote(proposalId, support);
  res.json({ message: 'Vote cast successfully' });
});

app.post('/api/astralplane-governance/execute', async (req, res) => {
  const proposalId = req.body.proposalId;
  await AstralPlaneGovernance.execute(proposalId);
  res.json({ message: 'Proposal executed successfully' });
});

app.listen(3001, () => {
  console.log('AstralPlane Governance API listening on port 3001');
});
