const { GraphQLObjectType, GraphQLString, GraphQLInt, GraphQLBoolean, GraphQLList } = require('graphql');
const { TransactionType } = require('../types');
const { getTransaction } = require('../blockchain');

const transactionResolver = {
  Query: {
    transaction: async (_, { hash }) => {
      const transaction = await getTransaction(hash);

      return transaction;
    },
  },
};

module.exports = transactionResolver;
