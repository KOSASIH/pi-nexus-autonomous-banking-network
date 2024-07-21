// compile-contract.js
const solc = require('solidity-compiler');

const contractSource = fs.readFileSync(
  'identity-verification-contract.sol',
  'utf8',
);
const compiledContract = solc.compile(
  contractSource,
  'identityVerificationContract',
);

fs.writeFileSync(
  'identity-verification-contract.json',
  JSON.stringify(compiledContract),
);
