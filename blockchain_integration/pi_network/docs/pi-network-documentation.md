# Pi Network Integration Documentation

## Setup

1.1. Import the necessary libraries:

```javascript
const PiNetwork = require("pi-network-javascript");
const walletIntegration = require("./pi-network-wallet-integration");
const rateLimiter = require("./pi-network-rate-limiter");
```

1.2. Set up the Pi Network API connection:

```javascript
const piNetwork = new PiNetwork({
  network: "mainnet", // or 'testnet'
  apiKey: "YOUR_API_KEY",
});
```

## Wallet Integration

2.1. Get the balance:

```javascrip
async function getBalance() {
  const balance = await walletIntegration.getBalance();
  console.log(`Your Pi balance: ${balance}`);
}
```

2.2. Send Pi:

```javascrip
async function sendPi(recipient, amount) {
  const transaction = await walletIntegration.sendPi(recipient, amount);
  console.log(`Transaction ID: ${transaction.id}`);
}
```

2.3. Receive Pi:

javascript

Open In Editor
Edit
Run
Copy code
async function receivePi() {
const transactions = await walletIntegration.receivePi();
console.log(`Received Pi transactions: ${transactions}`);
}

## Rate Limiter

3.1. Implement rate limiting for Pi Network API requests:

````javascript
async function makeApiRequest(endpoint, params) {
  const limit = await limiter.get();
  if (limit.remaining === 0) {
    console.log('Rate limit exceeded. Waiting for 1 hour...');
    await new Promise(resolve => setTimeout(resolve, 3600000));
  }

  try {
    const response = await piNetwork.api.request(endpoint, params);
    return response;
  } catch (error) {
    console.error('API request failed:', error);
  }
}

## Consensus Algorithm

4.1. Implement the SCP algorithm:

```javascrip
async function participateInConsensus() {
  // Create a list of quorum slices
  const quorumSlices = trustedNodes.map(node => ({
    validators: [node],
    value: 1.0
  }));

  // Participate in the consensus process
  const result = await piNetwork.consensus.propose({
    quorumSlices,
    value: 1.0
  });

  console.log('Consensus result:', result);
}
````

4.2. Run the consensus algorithm every 10 seconds:

```javascrip
setInterval(participateInConsensus, 10000);
```

4.3. Start the server:

```javascrip
const app = require('express')();

// Expose the rate-limited API request function
app.get('/api/rate-limited-request', (req, res) => {
  rateLimiter.makeApiRequest('/v1/account/balance', {})
    .then(response => res.json(response))
    .catch(error => res.status(500).json({ error }));
});

app.listen(3000, () => {
  console.log('Server listening on port 3000');
});
```

## Conclusion

The provided documentation covers all the necessary features of the Pi Network integration, including wallet integration, rate limiting, consensus algorithm, and server setup. This documentation will help users understand how to use the integration effectively and contribute to its development.
