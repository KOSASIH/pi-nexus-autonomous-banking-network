import { Injectable } from '@nestjs/common';
import { KycModel } from './kyc.model';
import { KycValidation } from './kyc.validation';
import { PiNetworkApi } from '../pi-network-api/pi-network-api';

@Injectable()
export class KycService {
  constructor(private readonly kycModel: KycModel, private readonly kycValidation: KycValidation, private readonly piNetworkApi: PiNetworkApi) {}

  async verifyIdentity(userId: string, identity: any) {
    const user = await this.kycModel.findById(userId);
    if (!user) {
      throw new Error('User not found');
    }
    const isValid = await this.kycValidation.validateIdentity(identity);
    if (!isValid) {
      throw new Error('Invalid identity');
    }
    const piNetworkResponse = await this.piNetworkApi.createPayment(user.piNetworkAddress, 1, 'KYC verification');
    if (!piNetworkResponse) {
      throw new Error('Pi Network API error');
    }
    user.kycVerified = true;
    await user.save();
    return user;
  }
  }
