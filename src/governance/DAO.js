class DAO {
    constructor() {
        this.proposals = [];
    }

    createProposal(description) {
        const proposal = { description, votes: 0 };
        this.proposals.push(proposal);
        return proposal;
    }

    voteOnProposal(index) {
        if (this.proposals[index]) {
            this.proposals[index].votes += 1;
        }
    }

    getProposalResults() {
        return this.proposals.map(p => ({
            description: p.description,
            votes: p.votes,
        }));
    }
}

export default new DAO();
