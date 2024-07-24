import * as MongoDB from 'mongodb';

class AstralPlaneDatabase {
  constructor() {
    this.client = new MongoDB.MongoClient('mongodb://localhost:27017', { useNewUrlParser: true, useUnifiedTopology: true });
    this.db = this.client.db('astralplane');
  }

  async createAsset(asset) {
    const collection = this.db.collection('assets');
    await collection.insertOne(asset);
  }

  async getAsset(assetId) {
    const collection = this.db.collection('assets');
    const asset = await collection.findOne({ _id: assetId });
    return asset;
  }

  async updateAsset(assetId, updates) {
    const collection = this.db.collection('assets');
    await collection.updateOne({ _id: assetId }, { $set: updates });
  }

  async deleteAsset(assetId) {
    const collection = this.db.collection('assets');
    await collection.deleteOne({ _id: assetId });
  }

  async getAssets(filter = {}) {
    const collection = this.db.collection('assets');
    const assets = await collection.find(filter).toArray();
    return assets;
  }

  async createUser(user) {
    const collection = this.db.collection('users');
    await collection.insertOne(user);
  }

  async getUser(userId) {
    const collection = this.db.collection('users');
    const user = await collection.findOne({ _id: userId });
    return user;
  }

  async updateUser(userId, updates) {
    const collection = this.db.collection('users');
    await collection.updateOne({ _id: userId }, { $set: updates });
  }

  async deleteUser(userId) {
    const collection = this.db.collection('users');
    await collection.deleteOne({ _id: userId });
  }

  async getTransactions(filter = {}) {
    const collection = this.db.collection('transactions');
    const transactions = await collection.find(filter).toArray();
    return transactions;
  }

  async createTransaction(transaction) {
    const collection = this.db.collection('transactions');
    await collection.insertOne(transaction);
  }

  async close() {
    await this.client.close();
  }
}

export default AstralPlaneDatabase;
