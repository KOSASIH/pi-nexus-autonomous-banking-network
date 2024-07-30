import { Injectable } from '@nestjs.common';
import { TokenRepository } from '../infrastructure/database/repository';
import { SidraToken } from '../contracts/SidraToken.sol';

@Injectable()
export class TokenService {
  constructor(private readonly tokenRepository: TokenRepository) {}

  async getTokenInfo(): Promise<TokenInfo> {
    // Call the SidraToken contract to get the token info
    const sidraToken = new SidraToken();
    const info = await sidraToken.getInfo();
    return info;
  }

  async mintToken(mintData: MintData): Promise<Token> {
    // Call the SidraToken contract to mint tokens
    const sidraToken = new SidraToken();
    const result = await sidraToken.mint(mintData);
    return result;
  }

  async burnToken(burnData: BurnData): Promise<Token> {
    // Call the SidraToken contract to burn tokens
    const sidraToken = new SidraToken();
    const result = await sidraToken.burn(burnData);
    return result;
  }
}
