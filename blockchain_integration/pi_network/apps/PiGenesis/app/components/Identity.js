import { useContractFunction, useEthers, useCall } from '@usedapp/core';
import { Contract } from '@ethersproject/contracts';
import { abi as uPortABI } from '@uport/sdk/build/abi/uPort.json';

const uPortAddress = '0x...'; // Replace with the uPort contract address

const uPortInterface = new ethers.utils.Interface(uPortABI);

const uportContract = new Contract(
  uPortAddress,
  uPortInterface,
  ethers.getDefaultProvider('mainnet')
);

function useuPort() {
  const { chainId } = useEthers();

  const [identity, setIdentity] = useState<Identity | null>(null);

  const { send: createIdentity } = useContractFunction(uportContract, 'createIdentity', {
    transactionName: 'Create Identity'
  });

  const { value: createIdentityValue } = useCall(
    {
      contract: uportContract,
      method: 'createIdentity',
      args: []
    },
    {
      chainId
    }
  );

  useEffect(() => {
    if (createIdentityValue) {
      setIdentity(createIdentityValue);
    }
  }, [createIdentityValue]);

  return { createIdentity, identity };
}

export default useuPort;
