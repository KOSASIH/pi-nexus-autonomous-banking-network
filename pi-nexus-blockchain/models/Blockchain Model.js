import { Model, DataTypes } from 'sequelize';
import { sequelize } from '../database';
import { NexusModel } from './NexusModel';

class BlockchainModel extends Model {
  static init(sequelize) {
    super.init({
      id: {
        type: DataTypes.UUID,
        defaultValue: DataTypes.UUIDV4,
        primaryKey: true,
      },
      networkId: {
        type: DataTypes.STRING,
        allowNull: false,
      },
      chainId: {
        type: DataTypes.STRING,
        allowNull: false,
      },
      blockNumber: {
        type: DataTypes.BIGINT,
        allowNull: false,
        defaultValue: 0,
      },
    }, {
      sequelize,
      modelName: 'Blockchain',
      tableName: 'blockchains',
      timestamps: true,
      underscored: true,
    });
  }

  static associate(models) {
    this.hasMany(models.Nexus, { foreignKey: 'blockchainId', onDelete: 'CASCADE' });
  }

  async updateBlockNumber() {
    const web3 = new Web3(new Web3.providers.HttpProvider(`https://mainnet.infura.io/v3/YOUR_PROJECT_ID`));
    const blockNumber = await web3.eth.getBlockNumber();
    this.blockNumber = blockNumber;
    await this.save();
  }
}

export default BlockchainModel;
