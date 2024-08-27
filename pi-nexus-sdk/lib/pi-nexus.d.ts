interface PiNexusOptions {
  apiUrl: string;
  apiKey: string;
}

class PiNexus {
  constructor(options: PiNexusOptions);

  getWallets(): Promise<Wallet[]>;
  getWallet(walletId: string): Promise<Wallet>;
  createWallet(walletData: WalletData): Promise<Wallet>;

  getTransactions(): Promise<Transaction[]>;
  getTransaction(transactionId: string): Promise<Transaction>;
  createTransaction(transactionData: TransactionData): Promise<Transaction>;

  getContracts(): Promise<Contract[]>;
  getContract(contractId: string): Promise<Contract>;
  createContract(contractData: ContractData): Promise<Contract>;
}

export default PiNexus;
