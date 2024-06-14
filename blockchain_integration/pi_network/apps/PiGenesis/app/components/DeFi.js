import { useContractFunction, useEthers, useCall } from '@usedapp/core';
import { Contract } from '@ethersproject/contracts';
import { abi as CompoundABI } from '@compound-finance/compound-js/build/abi/Comptroller.json';

const CompoundAddress = '0x3d9819210A31b4961b30EF54bE2aeD5B1F8c71C1';

const CompoundInterface = new ethers.utils.Interface(CompoundABI);

const compoundContract = new Contract(
  CompoundAddress,
  CompoundInterface,
  ethers.getDefaultProvider('mainnet')
);

function useCompound() {
  const { chainId } = useEthers();

  const [borrowRate, setBorrowRate] = useState<BigNumber | null>(null);
  const [supplyRate, setSupplyRate] = useState<BigNumber | null>(null);

  const { value: borrowRateValue } = useCall(
    {
      contract: compoundContract,
      method: 'borrowRatePerBlock',
      args: []
    },
    {
      chainId
    }
  );

  const { value: supplyRateValue } = useCall(
    {
      contract: compoundContract,
      method: 'upplyRatePerBlock',
      args: []
    },
    {
      chainId
    }
  );

  useEffect(() => {
    if (borrowRateValue) {
      setBorrowRate(borrowRateValue);
    }

    if (supplyRateValue) {
      setSupplyRate(supplyRateValue);
    }
  }, [borrowRateValue, supplyRateValue]);

  return { borrowRate, supplyRate };
}

export default useCompound;
