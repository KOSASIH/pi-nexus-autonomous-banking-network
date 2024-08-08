const piNetworkUtils = require('./utils/pi_network_utils');
const piTokenUtils = require('./utils/pi_token_utils');
const piNetworkGovernanceUtils = require('./utils/pi_network_governance_utils');

async function main() {
  try {
    // Get node count
    const nodeCount = await piNetworkUtils.getNodeCount();
    console.log(`Node count: ${nodeCount}`);

    // Get PI token balance
    const piTokenBalance = await piTokenUtils.getPiTokenBalance('0x...' /* your Ethereum address */);
    console.log(`PI token balance: ${piTokenBalance}`);

    // Get proposal count
    const proposalCount = await piNetworkGovernanceUtils.getProposalCount();
    console.log(`Proposal count: ${proposalCount}`);

    // Vote on proposal
    await piNetworkGovernanceUtils.voteOnProposal(0, true);
    console.log('Voted on proposal');
  } catch (error) {
    console.error(error);
  }
}

main();
