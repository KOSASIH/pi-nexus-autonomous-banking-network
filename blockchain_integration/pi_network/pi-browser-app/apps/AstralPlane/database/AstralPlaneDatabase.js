import * as MongoDB from 'ongodb';

class AstralPlaneDatabase {
  constructor() {
    this.client = new MongoDB.MongoClient('mongodb://localhost:27017', { useNewUrlParser: true, useUnifiedTopology: true });
    this.db = this.client.db('astralplane');
    this.assetsCollection = this.db.collection('assets');
  }

  async getAssets() {
    const assets = await this.assetsCollection.find().toArray();
    return assets;
  }

  async getAsset(id) {
    const asset = await this.assetsCollection.findOne({ _id: id });
    return asset;
  }

  async createAsset(asset) {
    await this.assetsCollection.insertOne(asset);
  }

  async updateAsset(id, asset) {
    await this.assetsCollection.updateOne({ _id: id }, { $set: asset });
  }

  async deleteAsset(id) {
    await this.assetsCollection.deleteOne({ _id: id });
  }
}

export default AstralPlaneDatabase;
