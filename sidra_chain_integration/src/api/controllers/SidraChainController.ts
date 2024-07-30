import { Controller, Get, Post, Body, Req, Res } from '@nestjs/common';
import { SidraChainService } from '../services/SidraChainService';
import { SidraChainTransactionRepository } from '../infrastructure/database/repository';
import { SendMessage } from '../infrastructure/messaging/rabbitmq';
import { Roles } from '../security/roles.decorator';
import { RolesGuard } from '../security/roles.guard';

@Controller('sidra-chain')
export class SidraChainController {
  constructor(
    private readonly sidraChainService: SidraChainService,
    private readonly sidraChainTransactionRepository: SidraChainTransactionRepository,
  ) {}

  @Get()
  @Roles('admin')
  async getSidraChainInfo(@Req() req: Request, @Res() res: Response) {
    const info = await this.sidraChainService.getSidraChainInfo();
    return res.json(info);
  }

  @Post('transfer')
  @Roles('user')
  async transfer(@Body() transferData: TransferData, @Req() req: Request, @Res() res: Response) {
    const result = await this.sidraChainService.transfer(transferData);
    await this.sidraChainTransactionRepository.save(result);
    await SendMessage(`Transaction ${result.transactionHash} has been processed`);
    return res.json(result);
  }

  @Get('transactions')
  @Roles('admin')
  async getTransactions(@Req() req: Request, @Res() res: Response) {
    const transactions = await this.sidraChainTransactionRepository.getTransactions();
    return res.json(transactions);
  }

  @Get('transaction/:transactionHash')
  @Roles('user')
  async getTransaction(@Param('transactionHash') transactionHash: string, @Req() req: Request, @Res() res: Response) {
    const transaction = await this.sidraChainTransactionRepository.getTransaction(transactionHash);
    if (!transaction) {
      throw new NotFoundException(`Transaction ${transactionHash} not found`);
    }
    return res.json(transaction);
  }
  }
