const { expect } = require('chai');
const Web3 = require('web3');

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

const Governance = artifacts.require('Governance');

contract('Governance', () => {
    let governance;

    beforeEach(async () => {
        governance = await Governance.new();
    });

    it('should create a proposal', async () => {
        const proposalId = await governance.createProposal('This is a test proposal');
        expect(proposalId).to.be.a('number');
    });

    it('should vote on a proposal', async () => {
        const proposalId = await governance.createProposal('This is a test proposal');
        await governance.vote(proposalId, true);
        const vote = await governance.getVote(proposalId, '0x...VoterAddress...');
        expect(vote).to.be.true;
    });

    it('should execute a proposal', async () => {
        const proposalId = await governance.createProposal('This is a test proposal');
        await governance.vote(proposalId, true);
        await governance.executeProposal(proposalId);
        const executed = await governance.getProposalExecuted(proposalId);
        expect(executed).to.be.true;
    });
});
