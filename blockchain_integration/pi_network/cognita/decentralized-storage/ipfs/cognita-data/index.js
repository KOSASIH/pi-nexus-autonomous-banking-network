import { IPFS } from 'ipfs';
import { CID } from 'ultiformats/cid';

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

async function addDataToIPFS(data) {
  const file = new File([data], 'cognita-data.txt');
  const added = await ipfs.add(file);
  return added.cid.toString();
}

async function getDataFromIPFS(cid) {
  const cidObj = CID.parse(cid);
  const data = await ipfs.cat(cidObj);
  return data.toString();
}

export { addDataToIPFS, getDataFromIPFS };
