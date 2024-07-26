import Web3 from 'web3';
import { TokenContract } from './contracts/TokenContract';

class UserManager {
  constructor(tokenContract) {
    this.tokenContract = tokenContract;
    this.users = {};
  }

  async createUser(address) {
    const user = {
      address: address,
      balance: await this.tokenContract.balanceOf(address),
    };

    this.users[address] = user;

    console.log(`User created: ${address} - balance: ${user.balance}`);
  }

  async getUser(address) {
    return this.users[address];
  }

  async updateUserBalance(address, amount) {
    const user = this.users[address];
    user.balance += amount;

    console.log(`User balance updated: ${address} - balance: ${user.balance}`);
  }
}

export { UserManager };
