// tests/node.test.js

const { Node } = require('../node');
const { expect } = require('chai');

describe('Node', () => {
  let node;

  beforeEach(() => {
    node = new Node();
  });

  it('should create a new node with a unique ID', () => {
    expect(node.id).to.be.a('string');
    expect(node.id).to.have.lengthOf(36);
  });

  it('should connect to other nodes', async () => {
    const otherNode = new Node();
    await node.connect(otherNode);
    expect(node.peers).to.include(otherNode);
  });

  it('should disconnect from other nodes', async () => {
    const otherNode = new Node();
    await node.connect(otherNode);
    await node.disconnect(otherNode);
    expect(node.peers).not.to.include(otherNode);
  });
});
