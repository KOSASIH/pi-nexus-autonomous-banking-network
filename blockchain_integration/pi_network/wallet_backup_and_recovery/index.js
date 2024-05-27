import { Web3 } from "web3";
import { ethers } from "ethers";
import { WalletBackupAndRecovery } from "./WalletBackupAndRecovery";
import { WalletBackupAndRecoveryController } from "./WalletBackupAndRecoveryController";

const web3 = new Web3(
  new Web3.providers.HttpProvider(
    "https://mainnet.infura.io/v3/YOUR_PROJECT_ID",
  ),
);
const wallet = new ethers.Wallet("0x1234567890abcdef", web3);

const walletBackupAndRecovery = new WalletBackupAndRecovery(
  "0xWALLET_BACKUP_AND_RECOVERY_CONTRACT_ADDRESS",
  web3,
);
const walletBackupAndRecoveryController = new WalletBackupAndRecoveryController(
  walletBackupAndRecovery,
  wallet,
);

export { walletBackupAndRecovery, walletBackupAndRecoveryController };
