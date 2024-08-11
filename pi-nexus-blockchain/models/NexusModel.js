import { Model, DataTypes } from 'sequelize';
import { sequelize } from '../database';
import { BlockchainModel } from './BlockchainModel';

class NexusModel extends Model {
  static init(sequelize) {
    super.init({
      id: {
        type: DataTypes.UUID,
        defaultValue: DataTypes.UUIDV4,
        primaryKey: true,
      },
      address: {
        type: DataTypes.STRING,
        allowNull: false,
        unique: true,
      },
      balance: {
        type: DataTypes.BIGINT,
        allowNull: false,
        defaultValue: 0,
      },
      blockchainId: {
        type: DataTypes.UUID,
        references: {
          model: BlockchainModel,
          key: 'id',
        },
      },
    }, {
      sequelize,
      modelName: 'Nexus',
      tableName: 'nexus',
      timestamps: true,
      underscored: true,
    });
  }

  static associate(models) {
    this.belongsTo(models.Blockchain, { foreignKey: 'blockchainId', onDelete: 'CASCADE' });
  }

  async deposit(amount) {
    this.balance += amount;
    await this.save();
  }

  async withdraw(amount) {
    if (this.balance < amount) {
      throw new Error('Insufficient balance');
    }
    this.balance -= amount;
    await this.save();
  }

  async transfer(to, amount) {
    if (this.balance < amount) {
      throw new Error('Insufficient balance');
    }
    const toNexus = await NexusModel.findOne({ where: { address: to } });
    if (!toNexus) {
      throw new Error('Recipient not found');
    }
    this.balance -= amount;
    toNexus.balance += amount;
    await Promise.all([this.save(), toNexus.save()]);
  }
}

export default NexusModel;
