// models/Payment.js
import { Model, DataTypes } from 'sequelize';
import { PaymentGatewayContract } from '../contracts/PaymentGatewayContract';

class Payment extends Model {
  static init(sequelize) {
    super.init(
      {
        id: {
          type: DataTypes.UUID,
          primaryKey: true,
          defaultValue: uuidv4(),
        },
        amount: {
          type: DataTypes.DECIMAL(18, 8),
          allowNull: false,
        },
        currency: {
          type: DataTypes.STRING,
          allowNull: false,
        },
        payer: {
          type: DataTypes.UUID,
          references: {
            model: 'Merchant',
            key: 'id',
          },
        },
        payee: {
          type: DataTypes.UUID,
          references: {
            model: 'Merchant',
            key: 'id',
          },
        },
        transactionHash: {
          type: DataTypes.STRING,
          allowNull: false,
        },
        status: {
          type: DataTypes.STRING,
          allowNull: false,
          defaultValue: 'pending',
        },
      },
      {
        sequelize,
        modelName: 'Payment',
        tableName: 'payments',
        timestamps: true,
      }
    );

    this.addHook('afterCreate', async (payment) => {
      try {
        const transactionHash = await PaymentGatewayContract.methods
          .processPayment(payment.payer, payment.payee, payment.amount)
          .send({ from: payment.payer });
        payment.update({ transactionHash, status: 'processed' });
      } catch (error) {
        payment.update({ status: 'failed' });
      }
    });
  }

  async setStatus(status) {
    this.update({ status });
  }
}

export default Payment;
