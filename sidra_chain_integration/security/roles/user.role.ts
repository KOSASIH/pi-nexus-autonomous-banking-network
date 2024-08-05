import { Role } from '../roles.decorator';

@Role('user')
export class UserRole {
  canReadSidraChainInfo(): boolean {
    return false;
  }

  canTransferTokens(): boolean {
    return true;
  }

  canViewTransactions(): boolean {
    return true;
  }
}
