import { Controller, Get, Post, Body, Req, Res } from '@nestjs/common';
import { TokenService } from '../services/TokenService';
import { TokenRepository } from '../infrastructure/database/repository';
import { Roles } from '../security/roles.decorator';
import { RolesGuard } from '../security/roles.guard';
import { AdminRole } from '../security/roles/admin.role';
import { UserRole } from '../security/roles/user.role';

@Controller('token')
export class TokenController {
  constructor(private readonly tokenService: TokenService, private readonly tokenRepository: TokenRepository) {}

  @Get()
  @Roles('admin')
  async getTokenInfo(@Req() req: Request, @Res() res: Response) {
    const info = await this.tokenService.getTokenInfo();
    return res.json(info);
  }

  @Post('mint')
  @Roles('admin')
  async mintToken(@Body() mintData: MintData, @Req() req: Request, @Res() res: Response) {
    const result = await this.tokenService.mintToken(mintData);
    await this.tokenRepository.save(result);
    return res.json(result);
  }

  @Post('burn')
  @Roles('admin')
  async burnToken(@Body() burnData: BurnData, @Req() req: Request, @Res() res: Response) {
    const result = await this.tokenService.burnToken(burnData);
    await this.tokenRepository.save(result);
    return res.json(result);
  }
  }
