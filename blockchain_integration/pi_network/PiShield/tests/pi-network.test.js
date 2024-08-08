// tests/pi-network.test.js

const { PiNetwork } = require('../pi-network');
const { expect } = require('chai');

describe('Pi Network', () => {
  let piNetwork;

  beforeEach(() => {
    piNetwork = new PiNetwork();
  });

  it('should create a new Pi Network with a set of nodes', () => {
    expect(piNetwork.nodes).to.be.an('array');
    expect(piNetwork.nodes).to.have.lengthOf(10);
  });

  it('should add new nodes to the network', async () => {
    const newNode = new Node();
    await piNetwork.addNode(newNode);
    expect(piNetwork.nodes).to.include(newNode);
  });

  it('should remove nodes from the network', async () => {
    const nodeToRemove = piNetwork.nodes[0];
    await piNetwork.removeNode(nodeToRemove);
    expect(piNetwork.nodes).not.to.include(nodeToRemove);
  });
});
