import { ethers } from 'ethers';

class PiNetworkContract {
  constructor(web3Provider) {
    this.piTokenContract = new ethers.Contract(
      '0x...PiTokenContractAddress...',
      [
        'function balanceOf(address) public view returns (uint256)',
        'function getStakingBalance(address) public view returns (uint256)',
        'function getLendingBalance(address) public view returns (uint256)',
        'function getGovernanceVotingPower(address) public view returns (uint256)',
      ],
      web3Provider
    );
  }

  async getPiTokenBalance() {
    return this.piTokenContract.balanceOf('0x...UserAddress...');
  }

  async getStakingBalance() {
    return this.piTokenContract.getStakingBalance('0x...UserAddress...');
  }

  async getLendingBalance() {
    return this.piTokenContract.getLendingBalance('0x...UserAddress...');
  }

  async getGovernanceVotingPower() {
    return this.piTokenContract.getGovernanceVotingPower('0x...UserAddress...');
  }
}

export default PiNetworkContract;
