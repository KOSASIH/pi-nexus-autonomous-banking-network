import { useContractFunction, useEthers, useCall } from '@usedapp/core';
import { Contract } from '@ethersproject/contracts';
import { abi as NuCypherABI } from '@nucypher/sdk/build/abi/NuCypher.json';

const NuCypherAddress = '0x...'; // Replace with the NuCypher contract address

const NuCypherInterface = new ethers.utils.Interface(NuCypherABI);

const nuCypherContract = new Contract(
  NuCypherAddress,
  NuCypherInterface,
  ethers.getDefaultProvider('mainnet')
);

function useNuCypher() {
  const { chainId } = useEthers();

  const [encryptedMessage, setEncryptedMessage] = useState<string | null>(null);

  const { value: encryptedMessageValue } = useCall(
    {
      contract: nuCypherContract,
      method: 'encryptMessage',
      args: ['Hello, world!']
    },
    {
      chainId
    }
  );

  useEffect(() => {
    if (encryptedMessageValue) {
      setEncryptedMessage(encryptedMessageValue);
    }
  }, [encryptedMessageValue]);

  return { encryptedMessage };
}

export default useNuCypher;
