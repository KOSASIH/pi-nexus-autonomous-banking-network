import { Swarm } from 'warm-js';
import { CID } from 'ultiformats/cid';

const swarm = new Swarm({
  bootstrap: [
    '/dnsaddr/bootstrap.libp2p.io/p2p/QmcZf59b...gU1ZjYZcYW3dwt',
    '/ip4/104.131.131.82/tcp/4001/p2p/QmaCpDMG...UtfsmvsqQLuvuJ',
  ],
});

async function addDataToSwarm(data) {
  const file = new File([data], 'cognita-data.txt');
  const added = await swarm.add(file);
  return added.cid.toString();
}

async function getDataFromSwarm(cid) {
  const cidObj = CID.parse(cid);
  const data = await swarm.get(cidObj);
  return data.toString();
}

export { addDataToSwarm, getDataFromSwarm };
