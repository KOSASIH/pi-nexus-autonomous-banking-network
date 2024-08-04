// kyc/kyc.module.js
import { Module } from '@cognita/core';
import KycController from './kyc.controller';
import KycService from './kyc.service';

@Module({
  controllers: [KycController],
  providers: [KycService],
})
export class KycModule {}
