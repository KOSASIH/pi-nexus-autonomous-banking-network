interface WalletOptions {
  apiUrl: string;
  apiKey: string;
}

class Wallet {
  constructor(options: WalletOptions);

  getWallets(): Promise<Wallet[]>;
  getWallet(walletId: string): Promise<Wallet>;
  createWallet(walletData: WalletData): Promise<Wallet>;
}

export default Wallet;
