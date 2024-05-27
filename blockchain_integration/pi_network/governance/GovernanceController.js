class GovernanceController {
  constructor(governance) {
    this.governance = governance;
  }

  async getProposals() {
    return await this.governance.getProposals();
  }

  async voteOnProposal(proposalId, vote) {
    return await this.governance.voteOnProposal(proposalId, vote);
  }

  async createProposal(proposal) {
    return await this.governance.createProposal(proposal);
  }
}

export default GovernanceController;
