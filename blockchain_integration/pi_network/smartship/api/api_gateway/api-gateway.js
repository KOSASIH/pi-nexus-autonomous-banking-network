const { ApolloServer } = require('@apollo/server');
const { graphqlUploadExpress } = require('graphql-upload');
const { makeExecutableSchema } = require('@graphql-tools/schema');
const { GraphQLJSON } = require('graphql-type-json');

const typeDefs = `
  type Query {
    getAccountBalance(accountId: ID!): Float!
    getTransactionHistory(accountId: ID!): [Transaction!]!
  }

  type Mutation {
    createAccount(name: String!, email: String!): Account!
    transferFunds(fromAccountId: ID!, toAccountId: ID!, amount: Float!): Transaction!
  }

  type Account {
    id: ID!
    name: String!
    email: String!
    balance: Float!
  }

  type Transaction {
    id: ID!
    fromAccountId: ID!
    toAccountId: ID!
    amount: Float!
    timestamp: DateTime!
  }
`;

const resolvers = {
  Query: {
    getAccountBalance: async (parent, { accountId }) => {
      // Call to blockchain API to retrieve account balance
      const balance = await blockchainApi.getAccountBalance(accountId);
      return balance;
    },
    getTransactionHistory: async (parent, { accountId }) => {
      // Call to blockchain API to retrieve transaction history
      const transactions = await blockchainApi.getTransactionHistory(accountId);
      return transactions;
    },
  },
  Mutation: {
    createAccount: async (parent, { name, email }) => {
      // Call to blockchain API to create a new account
      const account = await blockchainApi.createAccount(name, email);
      return account;
    },
    transferFunds: async (parent, { fromAccountId, toAccountId, amount }) => {
      // Call to blockchain API to transfer funds
      const transaction = await blockchainApi.transferFunds(fromAccountId, toAccountId, amount);
      return transaction;
    },
  },
};

const schema = makeExecutableSchema({ typeDefs, resolvers });

const server = new ApolloServer({
  schema,
  context: async ({ req }) => {
    // Authenticate and authorize requests
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];
    const user = await authenticateUser(token);
    return { user };
  },
  plugins: [graphqlUploadExpress()],
});

server.listen().then(({ url }) => {
  console.log(`API Gateway listening on ${url}`);
});
