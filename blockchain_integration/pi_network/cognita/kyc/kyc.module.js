import { Module } from '@nestjs/common';
import { KycController } from './kyc.controller';
import { KycService } from './kyc.service';
import { KycModel } from './kyc.model';
import { KycValidation } from './kyc.validation';
import { PassportModule } from '@nestjs/passport';
import { LocalStrategy } from './strategies/passport-local';
import { JwtStrategy } from './strategies/passport-jwt';
import { PiNetworkStrategy } from './strategies/passport-pi-network';

@Module({
  imports: [
    PassportModule,
  ],
  controllers: [KycController],
  providers: [KycService, KycModel, KycValidation, LocalStrategy, JwtStrategy, PiNetworkStrategy],
  exports: [KycService],
})
export class KycModule {}
