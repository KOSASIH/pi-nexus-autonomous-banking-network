import { useContractFunction, useEthers, useCall } from '@usedapp/core';
import { Contract } from '@ethersproject/contracts';
import { abi as AragonABI } from '@aragon/sdk/build/abi/Aragon.json';

const AragonAddress = '0x...'; // Replace with the Aragon contract address

const AragonInterface = new ethers.utils.Interface(AragonABI);

const aragonContract = new Contract(
  AragonAddress,
  AragonInterface,
  ethers.getDefaultProvider('mainnet')
);

function useAragon() {
  const { chainId } = useEthers();

  const [proposal, setProposal] = useState<Proposal | null>(null);

  const { send: createProposal } = useContractFunction(aragonContract, 'createProposal', {
    transactionName: 'Create Proposal'
  });

  const { value: createProposalValue } = useCall(
    {
      contract: aragonContract,
      method: 'createProposal',
      args: [proposal]
    },
    {
      chainId
    }
  );

  useEffect(() => {
    if (createProposalValue) {
      setProposal(createProposalValue);
    }
  }, [createProposalValue]);

  return { createProposal, proposal };
}

export default useAragon;
