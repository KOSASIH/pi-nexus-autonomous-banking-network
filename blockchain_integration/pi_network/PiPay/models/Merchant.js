// models/Merchant.js
import { Model, DataTypes } from 'sequelize';
import bcrypt from 'bcrypt';
import { v4 as uuidv4 } from 'uuid';
import { PaymentGatewayContract } from '../contracts/PaymentGatewayContract';

class Merchant extends Model {
  static init(sequelize) {
    super.init(
      {
        id: {
          type: DataTypes.UUID,
          primaryKey: true,
          defaultValue: uuidv4(),
        },
        name: {
          type: DataTypes.STRING,
          allowNull: false,
        },
        email: {
          type: DataTypes.STRING,
          unique: true,
          allowNull: false,
        },
        password: {
          type: DataTypes.STRING,
          allowNull: false,
        },
        address: {
          type: DataTypes.STRING,
          allowNull: false,
        },
        publicKey: {
          type: DataTypes.STRING,
          allowNull: false,
        },
        privateKey: {
          type: DataTypes.STRING,
          allowNull: false,
        },
      },
      {
        sequelize,
        modelName: 'Merchant',
        tableName: 'merchants',
        timestamps: true,
      }
    );

    this.addHook('beforeCreate', async (merchant) => {
      const salt = await bcrypt.genSalt(10);
      merchant.password = await bcrypt.hash(merchant.password, salt);
    });

    this.addHook('beforeUpdate', async (merchant) => {
      if (merchant.changed('password')) {
        const salt = await bcrypt.genSalt(10);
        merchant.password = await bcrypt.hash(merchant.password, salt);
      }
    });
  }

  async authenticate(password) {
    return bcrypt.compare(password, this.password);
  }

  async getPaymentGatewayAddress() {
    return PaymentGatewayContract.methods.getMerchantAddress(this.publicKey).call();
  }

  async getBalance() {
    return PaymentGatewayContract.methods.getBalance(this.publicKey).call();
  }
}

export default Merchant;
