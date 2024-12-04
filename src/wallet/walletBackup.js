// walletBackup.js

import { ethers } from 'ethers';
import fs from 'fs';
import path from 'path';

class WalletBackup {
    constructor() {
        this.walletFilePath = path.join(__dirname, 'walletBackup.json');
    }

    // Create a new wallet and generate a mnemonic phrase
    createWallet() {
        const wallet = ethers.Wallet.createRandom();
        const mnemonic = wallet.mnemonic.phrase;
        const privateKey = wallet.privateKey;

        // Save the wallet backup
        this.saveBackup(mnemonic, privateKey);
        console.log('Wallet created and backed up successfully.');
        return { mnemonic, privateKey };
    }

    // Save wallet backup to a file
    saveBackup(mnemonic, privateKey) {
        const backupData = {
            mnemonic,
            privateKey,
            timestamp: new Date().toISOString()
        };
        fs.writeFileSync(this.walletFilePath, JSON.stringify(backupData, null, 2));
    }

    // Recover wallet from mnemonic phrase
    recoverWallet(mnemonic) {
        const wallet = ethers.Wallet.fromMnemonic(mnemonic);
        console.log('Wallet recovered successfully.');
        return wallet;
    }

    // Load wallet backup from file
    loadBackup() {
        if (fs.existsSync(this.walletFilePath)) {
            const backupData = JSON.parse(fs.readFileSync(this.walletFilePath));
            return backupData;
        } else {
            throw new Error('No wallet backup found.');
        }
    }

    // Example usage
    exampleUsage() {
        // Create a new wallet
        const { mnemonic, privateKey } = this.createWallet();
        console.log('Mnemonic:', mnemonic);
        console.log('Private Key:', privateKey);

        // Load backup
        const backup = this.loadBackup();
        console.log('Loaded Backup:', backup);

        // Recover wallet
        const recoveredWallet = this.recoverWallet(backup.mnemonic);
        console.log('Recovered Wallet Address:', recoveredWallet.address);
    }
}

// Example usage
const walletBackup = new WalletBackup();
walletBackup.exampleUsage();

export default WalletBackup;
