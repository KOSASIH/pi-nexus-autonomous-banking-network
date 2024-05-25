import { WalletBackupAndRecovery } from './WalletBackupAndRecovery';

class WalletBackupAndRecoveryController {
  constructor(walletBackupAndRecovery, wallet) {
    this.walletBackupAndRecovery = walletBackupAndRecovery;
    this.wallet = wallet;
  }

  async backupWallet() {
    const seedPhrase = await this.walletBackupAndRecovery.generateSeedPhrase();
    const privateKey = await this.walletBackupAndRecovery.generatePrivateKey(seedPhrase);
    await this.walletBackupAndRecovery.backupWallet(seedPhrase, privateKey);
  }

  async recoverWallet(seedPhrase) {
    const privateKey = await this.walletBackupAndRecovery.recoverWallet(seedPhrase);
    this.wallet.importPrivateKey(privateKey);
  }
}

export { WalletBackupAndRecoveryController };
