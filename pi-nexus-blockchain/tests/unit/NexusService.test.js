import { NexusService } from '../../src/services/NexusService';
import { NexusModel } from '../../src/models/NexusModel';
import { BlockchainModel } from '../../src/models/BlockchainModel';
import sinon from 'sinon';
import { expect } from 'chai';

describe('NexusService', () => {
  let nexusService;
  let nexusModelStub;
  let blockchainModelStub;

  beforeEach(() => {
    nexusModelStub = sinon.stub(NexusModel, 'findOne');
    blockchainModelStub = sinon.stub(BlockchainModel, 'findOne');
    nexusService = new NexusService();
  });

  afterEach(() => {
    nexusModelStub.restore();
    blockchainModelStub.restore();
  });

  it('should get nexus by address', async () => {
    const address = '0x1234567890abcdef';
    const nexus = { id: 1, address, balance: 100 };
    nexusModelStub.resolves(nexus);

    const result = await nexusService.getNexusByAddress(address);
    expect(result).to.deep.equal(nexus);
  });

  it('should create new nexus', async () => {
    const address = '0x1234567890abcdef';
    const blockchain = { id: 1, networkId: 'mainnet' };
    blockchainModelStub.resolves(blockchain);

    const result = await nexusService.createNexus(address, blockchain.id);
    expect(result).to.have.property('id');
    expect(result).to.have.property('address', address);
    expect(result).to.have.property('balance', 0);
  });

  it('should deposit nexus tokens', async () => {
    const nexus = { id: 1, address: '0x1234567890abcdef', balance: 100 };
    nexusModelStub.resolves(nexus);

    const amount = 50;
    await nexusService.deposit(nexus.address, amount);
    expect(nexus.balance).to.equal(150);
  });

  it('should withdraw nexus tokens', async () => {
    const nexus = { id: 1, address: '0x1234567890abcdef', balance: 100 };
    nexusModelStub.resolves(nexus);

    const amount = 50;
    await nexusService.withdraw(nexus.address, amount);
    expect(nexus.balance).to.equal(50);
  });

  it('should transfer nexus tokens', async () => {
    const fromNexus = { id: 1, address: '0x1234567890abcdef', balance: 100 };
    const toNexus = { id: 2, address: '0xabcdef1234567890', balance: 50 };
    nexusModelStub.onCall(0).resolves(fromNexus);
    nexusModelStub.onCall(1).resolves(toNexus);

    const amount = 20;
    await nexusService.transfer(fromNexus.address, toNexus.address, amount);
    expect(fromNexus.balance).to.equal(80);
    expect(toNexus.balance).to.equal(70);
  });
});
