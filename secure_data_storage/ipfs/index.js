const IPFS = require('ipfs-http-client');
const ipfs = new IPFS({ host: 'ipfs.infura.io', port: 5001, protocol: 'https' });

const ipfsStorage = async (req, res) => {
    const data = req.body.data;
    const added = await ipfs.add(data);
    res.json(added);
};

module.exports = ipfsStorage;
