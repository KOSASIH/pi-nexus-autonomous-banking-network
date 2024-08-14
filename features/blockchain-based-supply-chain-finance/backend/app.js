const express = require('express');
const app = express();
const { FabricClient } = require('fabric-client');
const { Chaincode } = require('fabric-chaincode');

const client = new FabricClient();
const chaincode = new Chaincode('supply_chain_finance', '1.0');
app.use(express.json());

// Connect to the Fabric network
client.connect('grpc://localhost:7051', { asLocalhost: true })
  .then(() => {
    console.log('Connected to Fabric network');
  })
  .catch((err) => {
    console.error('Error connecting to Fabric network:', err);
  });

// Define API endpoints
app.post('/createTrade', (req, res) => {
  const { buyer, seller, product, quantity, price } = req.body;
  const tradeId = uuidv4();

  // Create a new trade object
  const trade = {
    id: tradeId,
    buyer,
    seller,
    product,
    quantity,
    price,
    status: 'pending',
  };

  // Invoke the createTrade function on the chaincode
  client.invokeChaincode(chaincode, 'createTrade', [tradeId, buyer, seller, product, quantity, price])
    .then((result) => {
      res.json({ message: 'Trade created successfully' });
    })
    .catch((err) => {
      res.status(500).json({ error: 'Error creating trade' });
    });
});

app.put('/updateTrade', (req, res) => {
  const { tradeId, status } = req.body;

  // Invoke the updateTrade function on the chaincode
  client.invokeChaincode(chaincode, 'updateTrade', [tradeId, status])
    .then((result) => {
      res.json({ message: 'Trade updated successfully' });
    })
    .catch((err) => {
      res.status(500).json({ error: 'Error updating trade' });
    });
});

app.get('/getTrade', (req, res) => {
  const { tradeId } = req.query;

  // Invoke the getTrade function on the chaincode
  client.invokeChaincode(chaincode, 'getTrade', [tradeId])
    .then((result) => {
      res.json(result);
    })
    .catch((err) => {
      res.status(500).json({ error: 'Error getting trade' });
    });
});

app.listen(3000, () => {
  console.log('Server listening on port 3000');
});
