import { Controller, Get, Post, Body, Req, Res } from '@nestjs/common';
import { SidraChainService } from '../services/SidraChainService';

@Controller('sidra-chain')
export class SidraChainController {
  constructor(private readonly sidraChainService: SidraChainService) {}

  @Get()
  async getSidraChainInfo(@Req() req: Request, @Res() res: Response) {
    const info = await this.sidraChainService.getSidraChainInfo();
    return res.json(info);
  }

  @Post('transfer')
  async transfer(@Body() transferData: TransferData, @Req() req: Request, @Res() res: Response) {
    const result = await this.sidraChainService.transfer(transferData);
    return res.json(result);
  }

  //...
                 }
