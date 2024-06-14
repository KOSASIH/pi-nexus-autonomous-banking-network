import { useContractFunction, useEthers, useCall } from '@usedapp/core';
import { Contract } from '@ethersproject/contracts';
import { abi as StatusABI } from '@status/status-js/build/abi/Status.json';

const StatusAddress = '0x...'; // Replace with the Status contract address

const StatusInterface = new ethers.utils.Interface(StatusABI);

const statusContract = new Contract(
  StatusAddress,
  StatusInterface,
  ethers.getDefaultProvider('mainnet')
);

function useStatus() {
  const { chainId } = useEthers();

  const [message, setMessage] = useState<string | null>(null);
  const [encryptedMessage, setEncryptedMessage] = useState<string | null>(null);

  const { send: sendMessage } = useContractFunction(statusContract, 'sendMessage', {
    transactionName: 'Send Message'
  });

  const { value: sendMessageValue } = useCall(
    {
      contract: statusContract,
      method: 'sendMessage',
      args: [message]
    },
    {
      chainId
    }
  );

  useEffect(() => {
    if (sendMessageValue) {
      setEncryptedMessage(sendMessageValue);
    }
  }, [sendMessageValue]);

  return { sendMessage, encryptedMessage };
}

export default useStatus;
