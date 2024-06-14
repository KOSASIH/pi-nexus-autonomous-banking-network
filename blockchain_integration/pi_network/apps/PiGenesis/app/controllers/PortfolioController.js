import { NextFunction, Request, Response } from 'express';
import { BlockchainService } from '../blockchain-integration/BlockchainService';

class PortfolioController {
  async getPortfolio(req: Request, res: Response, next: NextFunction) {
    const blockchainService = new BlockchainService();
    const portfolio = await blockchainService.getPortfolio(req.user.address);
    res.json(portfolio);
  }

  async deposit(req: Request, res: Response, next: NextFunction) {
    const blockchainService = new BlockchainService();
    const { amount } = req.body;
    const txHash = await blockchainService.deposit(req.user.address, amount);
    res.json({ txHash });
  }

  async withdraw(req: Request, res: Response, next: NextFunction) {
    const blockchainService = new BlockchainService();
    const { amount } = req.body;
    const txHash = await blockchainService.withdraw(req.user.address, amount);
    res.json({ txHash });
  }
}

export default PortfolioController;
