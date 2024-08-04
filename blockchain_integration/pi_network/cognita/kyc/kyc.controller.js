import { Controller, Post, Body, Req, Res } from '@nestjs/common';
import { KycService } from './kyc.service';
import { KycValidation } from './kyc.validation';

@Controller('kyc')
export class KycController {
  constructor(private readonly kycService: KycService, private readonly kycValidation: KycValidation) {}

  @Post('verify')
  async verifyIdentity(@Body() identity: any, @Req() req: Request, @Res() res: Response) {
    try {
      const user = await this.kycService.verifyIdentity(req.user.id, identity);
      res.json(user);
    } catch (error) {
      res.status(400).json({ message: error.message });
    }
  }
  }
