import { useContractFunction, useEthers, useCall } from '@usedapp/core';
import { Contract } from '@ethersproject/contracts';
import { abi as IPFSABI } from 'ipfs-http-client/src/utils/abi.js';

const IPFSAddress = '0x...'; // Replace with the IPFS contract address

const IPFSInterface = new ethers.utils.Interface(IPFSABI);

const ipfsContract = new Contract(
  IPFSAddress,
  IPFSInterface,
  ethers.getDefaultProvider('mainnet')
);

function useIPFS() {
const { chainId } = useEthers();

  const [file, setFile] = useState<File | null>(null);
  const [ipfsHash, setIpfsHash] = useState<string | null>(null);

  const { send: addFile } = useContractFunction(ipfsContract, 'addFile', {
    transactionName: 'Add File'
  });

  const { value: addFileValue } = useCall(
    {
      contract: ipfsContract,
      method: 'addFile',
      args: [file]
    },
    {
      chainId
    }
  );

  useEffect(() => {
    if (addFileValue) {
      setIpfsHash(addFileValue);
    }
  }, [addFileValue]);

  return { addFile, ipfsHash };
}

export default useIPFS;
