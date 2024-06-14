import { NextFunction, Request, Response } from 'express';
import BlockchainService from './BlockchainService';

const blockchainMiddleware = async (req: Request, res: Response, next: NextFunction) => {
  const blockchainService = new BlockchainService();
  req.blockchain = blockchainService;
  next();
};

export default blockchainMiddleware;
