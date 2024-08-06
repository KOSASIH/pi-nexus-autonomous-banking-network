import { migrate } from '@onflow/flow-js-sdk';
import { readFileSync } from 'fs';
import { join } from 'path';

const MIGRATIONS_DIR = './migrations';
const DEPLOYMENT_ACCOUNT = '0x01';
const DEPLOYMENT_KEY = readFileSync(join(__dirname, '../keys/deployment-key.pem'), 'utf8');

async function migrateContract(contractName) {
  const migrationPath = join(MIGRATIONS_DIR, `${contractName}.cdc`);
  const code = readFileSync(migrationPath, 'utf8');
  const [txId, err] = await migrate({
    code,
    name: contractName,
    authorizer: DEPLOYMENT_ACCOUNT,
    payer: DEPLOYMENT_ACCOUNT,
    proposer: DEPLOYMENT_ACCOUNT,
    keys: [DEPLOYMENT_KEY],
  });

  if (err) {
    console.error(`Error migrating contract ${contractName}: ${err}`);
    return;
  }

  console.log(`Contract ${contractName} migrated with txId ${txId}`);
}

async function main() {
  await migrateContract('HealthRecords');
  await migrateContract('MedicalBills');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
