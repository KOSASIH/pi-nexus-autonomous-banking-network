import { BlockchainService } from '../../src/services/BlockchainService';
import { BlockchainModel } from '../../src/models/BlockchainModel';
import sinon from 'sinon';
import { expect } from 'chai';

describe('BlockchainService', () => {
  let blockchainService;
  let blockchainModelStub;

  beforeEach(() => {
    blockchainModelStub = sinon.stub(BlockchainModel, 'findOne');
    blockchainService = new BlockchainService();
  });

  afterEach(() => {
    blockchainModelStub.restore();
  });

  it('should get blockchain by networkId', async () => {
    const networkId = 'mainnet';
    const blockchain = { id: 1, networkId, chainId: '1' };
    blockchainModelStub.resolves(blockchain);

    const result = await blockchainService.getBlockchainByNetworkId(networkId);
    expect(result).to.deep.equal(blockchain);
  });

  it('should update blockchain block number', async () => {
    const blockchain = { id: 1, networkId: 'mainnet', chainId: '1', blockNumber: 100 };
    blockchainModelStub.resolves(blockchain);

    const newBlockNumber = 150;
    await blockchainService.updateBlockchainBlockNumber(blockchain.id, newBlockNumber);
    expect(blockchain.blockNumber).to.equal(newBlockNumber);
  });
});
