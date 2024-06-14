import { useContractFunction, useEthers, useCall } from '@usedapp/core';
import { Contract } from '@ethersproject/contracts';
import { abi as ZcashABI } from '@zcash/sdk/build/abi/Zcash.json';

const ZcashAddress = '0x...'; // Replace with the Zcash contract address

const ZcashInterface = new ethers.utils.Interface(ZcashABI);

const zcashContract = new Contract(
  ZcashAddress,
  ZcashInterface,
  ethers.getDefaultProvider('mainnet')
);

function useZcash() {
  const { chainId } = useEthers();

  const [shieldedBalance, setShieldedBalance] = useState<BigNumber | null>(null);

  const { value: shieldedBalanceValue } = useCall(
    {
      contract: zcashContract,
      method: 'getShieldedBalance',
      args: []
    },
    {
      chainId
    }
  );

  useEffect(() => {
    if (shieldedBalanceValue) {
      setShieldedBalance(shieldedBalanceValue);
    }
  }, [shieldedBalanceValue]);

  return { shieldedBalance };
}

export default useZcash;
