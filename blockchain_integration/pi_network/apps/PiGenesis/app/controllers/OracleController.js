import { NextFunction, Request, Response } from 'express';
import { BlockchainService } from '../blockchain-integration/BlockchainService';

class OracleController {
  async getData(req: Request, res: Response, next: NextFunction) {
    const blockchainService = new BlockchainService();
    const { key } = req.params;
    const data = await blockchainService.getData(key);
    res.json(data);
  }

  async updateData(req: Request, res: Response, next: NextFunction) {
    const blockchainService = new BlockchainService();
    const { key, value } = req.body;
    await blockchainService.updateData(key, value);
    res.json({ message: 'Data updated successfully' });
  }
}

export default OracleController;
