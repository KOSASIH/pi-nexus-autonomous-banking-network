// Eonix Smart Contracts
const EonixSmartContracts = {
  // Compiler
  compiler: {
    name: 'solc',
    version: '0.8.10',
  },
  // Runtime
  runtime: {
    name: 'EVM',
    version: '1.10.0',
  },
  // Contracts
  contracts: [
    {
      name: 'EonixToken',
      code: 'pragma solidity ^0.8.0; contract EonixToken { ... }',
    },
    {
      name: 'EonixMarket',
      code: 'pragma solidity ^0.8.0; contract EonixMarket { ... }',
    },
  ],
};
