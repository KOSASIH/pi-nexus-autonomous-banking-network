// piNetworkAnalytics.ts
import Web3 from 'web3';
import { PiCoinGovernanceContract } from '../contracts/governance/PiCoinGovernance.sol';

interface PiNetworkAnalytics {
  getNodeCount(): Promise<number>;
  getUserCount(): Promise<number>;
  getProposalCount(): Promise<number>;
}

class PiNetworkAnalyticsImpl implements PiNetworkAnalytics {
  private web3: Web3;
  private piCoinGovernanceContract: PiCoinGovernanceContract;

  constructor(web3: Web3, piCoinGovernanceContract: PiCoinGovernanceContract) {
    this.web3 = web3;
    this.piCoinGovernanceContract = piCoinGovernanceContract;
  }

  async getNodeCount(): Promise<number> {
    const nodeCount = await this.piCoinGovernanceContract.methods.getNodeCount().call();
    return nodeCount.toNumber();
  }

  async getUserCount(): Promise<number> {
    const userCount = await this.piCoinGovernanceContract.methods.getUserCount().call();
    return userCount.toNumber();
  }

  async getProposalCount(): Promise<number> {
    const proposalCount = await this.piCoinGovernanceContract.methods.getProposalCount().call();
    return proposalCount.toNumber();
  }
}

export { PiNetworkAnalytics, PiNetworkAnalyticsImpl };
