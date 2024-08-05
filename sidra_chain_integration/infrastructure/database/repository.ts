import { EntityRepository, Repository } from 'typeorm';
import { SidraChainTransaction } from './schema';

@EntityRepository(SidraChainTransaction)
export class SidraChainTransactionRepository extends Repository<SidraChainTransaction> {
  async getTransactions(): Promise<SidraChainTransaction[]> {
    return await this.find();
  }

  async getTransaction(transactionHash: string): Promise<SidraChainTransaction | undefined> {
    return await this.findOne({ where: { transactionHash } });
  }
}
