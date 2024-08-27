export const formatBlock = (block: any) => {
  return {
    number: block.number,
    hash: block.hash,
    transactions: block.transactions,
  };
};

export const formatTransaction = (transaction: any) => {
  return {
    hash: transaction.hash,
    from: transaction.from,
    to: transaction.to,
    value: transaction.value,
    contractAddress: transaction.contractAddress,
  };
};

export const formatContract = (contract: any) => {
  return {
    address: contract.address,
    bytecode: contract.bytecode,
    abi: contract.abi,
  };
};
