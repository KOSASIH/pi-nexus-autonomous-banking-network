import { Swarm } from 'warm-js';
import { CID } from 'ultiformats/cid';

class PeerManager {
  async connectToPeers() {
    const swarm = new Swarm({
      bootstrap: [
        '/dnsaddr/bootstrap.libp2p.io/p2p/QmcZf59b...gU1ZjYZcYW3dwt',
        '/ip4/104.131.131.82/tcp/4001/p2p/QmaCpDMG...UtfsmvsqQLuvuJ',
      ],
    });
    await swarm.connect();
  }

  async disconnectFromPeers() {
    const swarm = new Swarm({
      bootstrap: [
        '/dnsaddr/bootstrap.libp2p.io/p2p/QmcZf59b...gU1ZjYZcYW3dwt',
        '/ip4/104.131.131.82/tcp/4001/p2p/QmaCpDMG...UtfsmvsqQLuvuJ',
      ],
    });
    await swarm.disconnect();
  }
}

export default PeerManager;
