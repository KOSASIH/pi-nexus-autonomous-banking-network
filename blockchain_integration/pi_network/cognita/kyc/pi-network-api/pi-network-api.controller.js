import { Controller, Post, Body, Req, Res } from '@nestjs/common';
import { PiNetworkApiService } from './pi-network-api.service';

@Controller('pi-network-api')
export class PiNetworkApiController {
  constructor(private readonly piNetworkApiService: PiNetworkApiService) {}

  @Post('create-payment')
  async createPayment(@Body() payment: any, @Req() req: Request, @Res() res: Response) {
    try {
      const response = await this.piNetworkApiService.createPayment(payment.piNetworkAddress, payment.amount, payment.description);
      res.json(response);
    } catch (error) {
      res.status(400).json({ message: error.message });
    }
  }
}
