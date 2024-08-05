import { Model, Document } from 'ongoose';

export interface KycDocument extends Document {
  userId: string;
  kycVerified: boolean;
  piNetworkAddress: string;
}

export class KycModel extends Model<KycDocument> {
  static async findById(userId: string) {
    return this.findOne({ userId });
  }
}
