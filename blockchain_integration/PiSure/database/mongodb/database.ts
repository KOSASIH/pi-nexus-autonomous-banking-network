import { MongoClient } from 'mongodb';

const config = {
  url: process.env.MONGODB_URL,
  dbName: 'pi-sure',
};

const client = new MongoClient(config.url, { useNewUrlParser: true, useUnifiedTopology: true });

export default client.db(config.dbName);
