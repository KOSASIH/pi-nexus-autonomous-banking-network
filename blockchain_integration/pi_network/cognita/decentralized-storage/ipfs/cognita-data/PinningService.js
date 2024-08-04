import { IPFS } from 'ipfs';
import { CID } from 'ultiformats/cid';

class PinningService {
  async pin(cid) {
    const ipfs = new IPFS({
      repo: './ipfs/repo',
      config: {
        Addresses: {
          Swarm: [
            '/ip4/0.0.0.0/tcp/4001',
            '/ip6/::/tcp/4001',
          ],
        },
      },
    });
    await ipfs.pin.add(cid);
  }

  async unpin(cid) {
    const ipfs = new IPFS({
      repo: './ipfs/repo',
      config: {
        Addresses: {
          Swarm: [
            '/ip4/0.0.0.0/tcp/4001',
            '/ip6/::/tcp/4001',
          ],
        },
      },
    });
    await ipfs.pin.rm(cid);
  }
}

export default PinningService;
