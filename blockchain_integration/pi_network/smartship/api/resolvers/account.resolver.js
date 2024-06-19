const { GraphQLObjectType, GraphQLString, GraphQLInt, GraphQLBoolean, GraphQLList } = require('graphql');
const { AccountType } = require('../types');
const { createAccount, getAccountBalance, getTransactionHistory } = require('../blockchain');

const accountResolver = {
  Query: {
    account: async (_, { address }) => {
      const balance = await getAccountBalance(address);
      const transactionHistory = await getTransactionHistory(address);

      return {
        address,
        balance,
        transactionHistory,
      };
    },
  },
  Mutation: {
    createAccount: async (_, { address, initialBalance }) => {
      const receipt = await createAccount(address, initialBalance);

      return {
        address,
        initialBalance,
        transactionHash: receipt.transactionHash,
      };
    },
  },
};

module.exports = accountResolver;
