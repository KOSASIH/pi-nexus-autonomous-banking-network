import { deploy } from '@onflow/flow-js-sdk';
import { readFileSync } from 'fs';
import { join } from 'path';

const CONTRACTS_DIR = './contracts';
const DEPLOYMENT_ACCOUNT = '0x01';
const DEPLOYMENT_KEY = readFileSync(join(__dirname, '../keys/deployment-key.pem'), 'utf8');

async function deployContract(contractName) {
  const contractPath = join(CONTRACTS_DIR, `${contractName}.cdc`);
  const code = readFileSync(contractPath, 'utf8');
  const [txId, err] = await deploy({
    code,
    name: contractName,
    authorizer: DEPLOYMENT_ACCOUNT,
    payer: DEPLOYMENT_ACCOUNT,
    proposer: DEPLOYMENT_ACCOUNT,
    keys: [DEPLOYMENT_KEY],
  });

  if (err) {
    console.error(`Error deploying contract ${contractName}: ${err}`);
    return;
  }

  console.log(`Contract ${contractName} deployed with txId ${txId}`);
}

async function main() {
  await deployContract('HealthRecords');
  await deployContract('MedicalBills');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
