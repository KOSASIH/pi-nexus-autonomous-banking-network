const { FabricClient } = require('fabric-client');

class BlockchainIntegrationController {
  async deploySmartContract(req, res) {
    const { contractCode } = req.body;
    const client = new FabricClient();
    const contract = client.deployContract(contractCode);
    res.json({ contractId: contract.getId() });
  }
}
