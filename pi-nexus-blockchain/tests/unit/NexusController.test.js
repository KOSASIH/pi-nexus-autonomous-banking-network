import { NexusController } from '../../src/controllers/NexusController';
import { NexusService } from '../../src/services/NexusService';
import sinon from 'sinon';
import { expect } from 'chai';

describe('NexusController', () => {
  let nexusController;
  let nexusServiceStub;

  beforeEach(() => {
    nexusServiceStub = sinon.stub(NexusService, 'getNexusByAddress');
    nexusController = new NexusController();
  });

  afterEach(() => {
    nexusServiceStub.restore();
  });

  it('should get nexus by address', async () => {
    const address = '0x1234567890abcdef';
    const nexus = { id: 1, address, balance: 100 };
    nexusServiceStub.resolves(nexus);

    const req = { params: { address } };
    const res = { json: sinon.spy() };
    await nexusController.getNexusByAddress(req, res);
    expect(res.json.calledWith(nexus)).to.be.true;
  });

  it('should create new nexus', async () => {
    const address = '0x1234567890abcdef';
    const blockchainId = 1;
    const nexus = { id: 1, address, balance: 0 };
    nexusServiceStub.resolves(nexus);

    const req = { body: { address, blockchainId } };
    const res = { json: sinon.spy() };
    await nexusController.createNexus(req, res);
    expect(res.json.calledWith(nexus)).to.be.true;
  });
});
