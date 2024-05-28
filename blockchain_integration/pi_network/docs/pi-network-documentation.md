# Pi Network Integration Documentation

## Setup

1.1. Import the necessary libraries:

```javascript
1. const PiNetwork = require('pi-network-javascript');
2. const walletIntegration = require('./pi-network-wallet-integration');
3. const rateLimiter = require('./pi-network-rate-limiter');
```

1.2. Set up the Pi Network API connection:

```javascript
1. const piNetwork = new PiNetwork({
2.  network: 'mainnet', // or 'testnet'
3.  apiKey: 'YOUR_API_KEY'
4. });
```

## Wallet Integration

2.1. Get the balance:

```javascrip
1. async function getBalance() {
2.  const balance = await walletIntegration.getBalance();
3.  console.log(`Your Pi balance: ${balance}`);
4. }
```
  
2.2. Send Pi:

```javascrip
1. async function sendPi(recipient, amount) {
2.  const transaction = await walletIntegration.sendPi(recipient, amount);
3.  console.log(`Transaction ID: ${transaction.id}`);
4. }
```

2.3. Receive Pi:

```javascrip
1. async function receivePi() {
2.  const transactions = await walletIntegration.receivePi();
3.  console.log(`Received Pi transactions: ${transactions}`);
4. }

## Rate Limiter

3.1. Implement rate limiting for Pi Network API requests:

```javascript
1. async function makeApiRequest(endpoint, params) {
2.  const limit = await limiter.get();
3.  if (limit.remaining === 0) {
4.    console.log('Rate limit exceeded. Waiting for 1 hour...');
5.    await new Promise(resolve => setTimeout(resolve, 3600000));
6.  }
7. 
8.  try {
9.    const response = await piNetwork.api.request(endpoint, params);
10.    return response;
11.  } catch (error) {
12.    console.error('API request failed:', error);
13.  }
14. }

## Consensus Algorithm

4.1. Implement the SCP algorithm:

```javascrip
1. async function participateInConsensus() {
2.  // Create a list of quorum slices
3.  const quorumSlices = trustedNodes.map(node => ({
3.    validators: [node],
4.    value: 1.0
5. }));
6. 
7.  // Participate in the consensus process
8.  const result = await piNetwork.consensus.propose({
9.    quorumSlices,
10.    value: 1.0
11.  });
12. 
13.  console.log('Consensus result:', result);
14. }
```
  
4.2. Run the consensus algorithm every 10 seconds:

```javascrip
1. setInterval(participateInConsensus, 10000);
```
  
4.3. Start the server:

```javascrip
1. const app = require('express')();
2. 
3. // Expose the rate-limited API request function
4. app.get('/api/rate-limited-request', (req, res) => {
5.  rateLimiter.makeApiRequest('/v1/account/balance', {})
6.    .then(response => res.json(response))
7.    .catch(error => res.status(500).json({ error }));
8. });
9. 
10. app.listen(3000, () => {
11.  console.log('Server listening on port 3000');
12. });
```

## Conclusion

The provided documentation covers all the necessary features of the Pi Network integration, including wallet integration, rate limiting, consensus algorithm, and server setup. This documentation will help users understand how to use the integration effectively and contribute to its development.
