// src/backend/app.js
import express from 'express';
import { json } from 'body-parser';
import { createServer } from 'http';
import { ApolloServer } from 'apollo-server-express';
import { typeDefs } from './schema';
import { resolvers } from './resolvers';
import { PiTokenContract } from './contracts/PiTokenContract';
import { PaymentGatewayContract } from './contracts/PaymentGatewayContract';
import { Web3Provider } from './providers/Web3Provider';

const app = express();
app.use(json());

const web3Provider = new Web3Provider();
const piTokenContract = new PiTokenContract(web3Provider);
const paymentGatewayContract = new PaymentGatewayContract(web3Provider);

const apolloServer = new ApolloServer({
  typeDefs,
  resolvers,
  context: ({ req, res }) => ({
    req,
    res,
    piTokenContract,
    paymentGatewayContract,
  }),
});

apolloServer.applyMiddleware({ app, cors: false });

const httpServer = createServer(app);
apolloServer.installSubscriptionHandlers(httpServer);

const port = process.env.PORT || 4000;
httpServer.listen(port, () => {
  console.log(`Pi-Based Payment Gateway listening on port ${port}`);
});

// Web3 event listeners
piTokenContract.on('Transfer', (from, to, value) => {
  console.log(`Pi Token transfer: ${from} -> ${to} (${value} PI)`);
});

paymentGatewayContract.on('PaymentProcessed', (payer, payee, amount) => {
  console.log(`Payment processed: ${payer} -> ${payee} (${amount} PI)`);
});
