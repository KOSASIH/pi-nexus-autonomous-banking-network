import { User } from '../models/User';
import { Portfolio } from '../models/Portfolio';
import { BlockchainService } from './BlockchainService';

class PortfolioService {
  async getPortfolio(address) {
    const user = await User.findOne({ address });
    if (!user) {
      throw new Error('User not found');
    }
    const portfolio = await Portfolio.findOne({ userId: user._id });
    if (!portfolio) {
      throw new Error('Portfolio not found');
    }
    return portfolio;
  }

  async deposit(address, amount) {
    const blockchainService = new BlockchainService();
    const txHash = await blockchainService.sendTransaction(address, '0x...', amount);
    const user = await User.findOne({ address });
    if (!user) {
      throw new Error('User not found');
    }
    const portfolio = await Portfolio.findOne({ userId: user._id });
    if (!portfolio) {
      throw new Error('Portfolio not found');
    }
    portfolio.value += amount;
    await portfolio.save();
    return txHash;
  }

  async withdraw(address, amount) {
    const blockchainService = new BlockchainService();
    const txHash = await blockchainService.sendTransaction('0x...', address, amount);
    const user = await User.findOne({ address });
    if (!user) {
      throw new Error('User not found');
    }
    const portfolio = await Portfolio.findOne({ userId: user._id });
    if (!portfolio) {
      throw new Error('Portfolio not found');
    }
    portfolio.value -= amount;
    await portfolio.save();
    return txHash;
  }
}

export default PortfolioService;
