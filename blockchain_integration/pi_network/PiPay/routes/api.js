// routes/api.js
import express from 'express';
import { graphqlHTTP } from 'express-graphql';
import { ApolloServer } from 'apollo-server-express';
import { typeDefs } from '../schema';
import { resolvers } from '../resolvers';
import { authenticate } from '../middleware/authenticate';
import { rateLimit } from '../middleware/rateLimit';
import { Merchant } from '../models/Merchant';
import { Payment } from '../models/Payment';

const api = express.Router();

// GraphQL API
const apolloServer = new ApolloServer({
  typeDefs,
  resolvers,
  context: ({ req, res }) => ({
    req,
    res,
    Merchant,
    Payment,
  }),
});

api.use('/graphql', authenticate, rateLimit, graphqlHTTP((req, res) => ({
  schema: apolloServer.schema,
  graphiql: true,
  pretty: true,
})));

// REST API
api.get('/merchants', authenticate, async (req, res) => {
  const merchants = await Merchant.findAll();
  res.json(merchants);
});

api.get('/merchants/:id', authenticate, async (req, res) => {
  const merchant = await Merchant.findByPk(req.params.id);
  if (!merchant) {
    res.status(404).json({ error: 'Merchant not found' });
  } else {
    res.json(merchant);
  }
});

api.post('/payments', authenticate, async (req, res) => {
  const { amount, currency, payer, payee } = req.body;
  const payment = await Payment.create({ amount, currency, payer, payee });
  res.json(payment);
});

api.get('/payments', authenticate, async (req, res) => {
  const payments = await Payment.findAll();
  res.json(payments);
});

api.get('/payments/:id', authenticate, async (req, res) => {
  const payment = await Payment.findByPk(req.params.id);
  if (!payment) {
    res.status(404).json({ error: 'Payment not found' });
  } else {
    res.json(payment);
  }
});

export default api;
