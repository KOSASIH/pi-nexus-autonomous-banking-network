// piNetworkWallet.ts
import Web3 from 'web3';
import { PiCoinStakingContract } from '../contracts/staking/PiCoinStaking.sol';

interface PiNetworkWallet {
  address: string;
  balance: number;
  stakedBalance: number;
}

class PiNetworkWalletImpl implements PiNetworkWallet {
  private web3: Web3;
  private piCoinStakingContract: PiCoinStakingContract;

  constructor(address: string, web3: Web3, piCoinStakingContract: PiCoinStakingContract) {
    this.address = address;
    this.web3 = web3;
    this.piCoinStakingContract = piCoinStakingContract;
  }

  async getBalance(): Promise<number> {
    const balance = await this.web3.eth.getBalance(this.address);
    return balance.toNumber();
  }

  async getStakedBalance(): Promise<number> {
    const stakedBalance = await this.piCoinStakingContract.methods.getStakedBalance(this.address).call();
    return stakedBalance.toNumber();
  }
}

export { PiNetworkWallet, PiNetworkWalletImpl };
