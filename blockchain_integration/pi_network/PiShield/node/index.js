const { PiNetwork } = require('./pi_network');

class Node {
    async start() {
        const piNetwork = new PiNetwork();
        await piNetwork.connect();
        console.log('Node started');
    }
}

module.exports = Node;
