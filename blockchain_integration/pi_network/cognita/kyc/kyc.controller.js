// kyc/kyc.controller.js
import { Controller, Post, Body } from '@cognita/core';
import KycService from './kyc.service';

@Controller('kyc')
export class KycController {
  constructor(private readonly kycService: KycService) {}

  @Post('verify')
  async verifyKyc(@Body() userData: any) {
    const userId = '12345'; // Replace with actual user ID
    const result = await this.kycService.verifyKyc(userId, userData);
    return { result };
  }
}
