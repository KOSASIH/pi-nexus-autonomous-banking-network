import { useContractFunction, useEthers, useCall } from '@usedapp/core';
import { Contract } from '@ethersproject/contracts';
import { abi as IOTAABI } from 'iota/build/abi/IOTA.json';

const IOTAAddress = '0x...'; // Replace with the IOTA contract address

const IOTAInterface = new ethers.utils.Interface(IOTAABI);

const iotaContract = new Contract(
  IOTAAddress,
  IOTAInterface,
  ethers.getDefaultProvider('mainnet')
);

function useIOTA() {
  const { chainId } = useEthers();

  const [device, setDevice] = useState<Device | null>(null);

  const { send: requestDevice } = useContractFunction(iotaContract, 'requestDevice', {
    transactionName: 'Request Device'
  });

  const { value: requestDeviceValue } = useCall(
    {
      contract: iotaContract,
      method: 'requestDevice',
      args: [device]
    },
    {
      chainId
    }
  );

  useEffect(() => {
    if (requestDeviceValue) {
      setDevice(requestDeviceValue);
    }
  }, [requestDeviceValue]);

  return { requestDevice, device };
}

export default useIOTA;
