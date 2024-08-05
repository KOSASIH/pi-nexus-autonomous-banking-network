import { Injectable } from '@nestjs/common';
import { SidraChainTransactionRepository } from '../infrastructure/database/repository';
import { SidraChain } from '../contracts/SidraChain.sol';

@Injectable()
export class SidraChainService {
  constructor(private readonly sidraChainTransactionRepository: SidraChainTransactionRepository) {}

  async getSidraChainInfo(): Promise<SidraChainInfo> {
    // Call the SidraChain contract to get the chain info
    const sidraChain = new SidraChain();
    const info = await sidraChain.getInfo();
    return info;
  }

  async transfer(transferData: TransferData): Promise<SidraChainTransaction> {
    // Call the SidraChain contract to transfer tokens
    const sidraChain = new SidraChain();
    const result = await sidraChain.transfer(transferData);
    return result;
  }
}
