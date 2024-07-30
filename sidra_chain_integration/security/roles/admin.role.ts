import { Role } from '../roles.decorator';

@Role('admin')
export class AdminRole {
  canReadSidraChainInfo(): boolean {
    return true;
  }

  canTransferTokens(): boolean {
    return true;
  }

  canViewTransactions(): boolean {
    return true;
  }
}
