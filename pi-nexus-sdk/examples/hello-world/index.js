import PiNexus from '../lib/pi-nexus';

const piNexus = new PiNexus('https://api.pi-nexus.io', 'YOUR_API_KEY');

async function main() {
  try {
    const wallets = await piNexus.getWallets();
    console.log(wallets);

    const wallet = await piNexus.getWallet('0x1234567890abcdef');
    console.log(wallet);

    const transaction = await piNexus.createTransaction({
      from: '0x1234567890abcdef',
      to: '0x9876543210fedcba',
      amount: 1.0
    });
    console.log(transaction);

    const contract = await piNexus.createContract({
      bytecode: '0x1234567890abcdef',
      abi: ['function foo() public']
    });
    console.log(contract);
  } catch (err) {
    console.error(err);
  }
}

main();
