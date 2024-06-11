import { IPFS } from 'ipfs-http-client';
import { Swarm } from 'swarm-ethereum';

class DecentralizedStorage {
  private ipfs: IPFS;
  private swarm: Swarm;

  constructor() {
    this.ipfs = new IPFS({ host: 'ipfs.infura.io', port: 5001, protocol: 'https' });
    this.swarm = new Swarm();
  }

  async storeData(data: any): Promise<string> {
    const cid = await this.ipfs.add(data);
    const swarmHash = await this.swarm.upload(cid);
    return swarmHash;
  }

  async retrieveData(swarmHash: string): Promise<any> {
    const cid = await this.swarm.download(swarmHash);
    const data = await this.ipfs.cat(cid);
    return data;
  }
}

export default DecentralizedStorage;
