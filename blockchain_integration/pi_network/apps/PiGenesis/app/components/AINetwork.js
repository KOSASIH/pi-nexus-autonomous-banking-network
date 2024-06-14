import { useContractFunction, useEthers, useCall } from '@usedapp/core';
import { Contract } from '@ethersproject/contracts';
import { abi as SingularityNETABI } from 'singularitynet/build/abi/SingularityNET.json';

const SingularityNETAddress = '0x...'; // Replace with the SingularityNET contract address

const SingularityNETInterface = new ethers.utils.Interface(SingularityNETABI);

const singularityNETContract = new Contract(
  SingularityNETAddress,
  SingularityNETInterface,
  ethers.getDefaultProvider('mainnet')
);

function useSingularityNET() {
  const { chainId } = useEthers();

  const [service, setService] = useState<Service | null>(null);

  const { send: requestService } = useContractFunction(singularityNETContract, 'requestService', {
    transactionName: 'Request Service'
  });

  const { value: requestServiceValue } = useCall(
    {
      contract: singularityNETContract,
      method: 'requestService',
      args: [service]
    },
    {
      chainId
    }
  );

  useEffect(() => {
    if (requestServiceValue) {
      setService(requestServiceValue);
    }
  }, [requestServiceValue]);

  return { requestService, service };
}

export default useSingularityNET;
