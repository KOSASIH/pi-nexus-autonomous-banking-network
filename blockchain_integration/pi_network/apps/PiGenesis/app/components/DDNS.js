import { useContractFunction, useEthers, useCall } from '@usedapp/core';
import { Contract } from '@ethersproject/contracts';
import { abi as ENSABI } from 'ethers-ens/src/contracts/ENS.json';

const ENSAddress = '0x...'; // Replace with the ENS contract address

const ENSInterface = new ethers.utils.Interface(ENSABI);

const ensContract = new Contract(
  ENSAddress,
  ENSInterface,
  ethers.getDefaultProvider('mainnet')
);

function useENS() {
  const { chainId } = useEthers();

  const [domainName, setDomainName] = useState<string | null>(null);
  const [ipfsHash, setIpfsHash] = useState<string | null>(null);

  const { send: registerDomain } = useContractFunction(ensContract, 'register', {
    transactionName: 'Register Domain'
  });

  const { value: registerDomainValue } = useCall(
    {
      contract: ensContract,
      method: 'register',
      args: [domainName, ipfsHash]
    },
    {
      chainId
    }
  );

  useEffect(() => {
    if (registerDomainValue) {
      setIpfsHash(registerDomainValue);
    }
  }, [registerDomainValue]);

  return { registerDomain, ipfsHash };
}

export default useENS;
