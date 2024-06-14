import { useContractFunction, useEthers, useCall } from '@usedapp/core';
import { Contract } from '@ethersproject/contracts';
import { abi as OpenSeaABI } from '@opensea/sdk/build/abi/OpenSea.json';

const OpenSeaAddress = '0x495f947276749Ce646f68AC8e984445e443d7B9';

const OpenSeaInterface = new ethers.utils.Interface(OpenSeaABI);

const openSeaContract = new Contract(
  OpenSeaAddress,
  OpenSeaInterface,
  ethers.getDefaultProvider('mainnet')
);

function useOpenSea() {
  const { chainId } = useEthers();

  const [nft, setNft] = useState<NFT | null>(null);

  const { value: nftValue } = useCall(
    {
      contract: openSeaContract,
      method: 'getNFT',
      args: ['0x...'] // Replace with the NFT contract address
    },
    {
      chainId
    }
  );

  useEffect(() => {
    if (nftValue) {
      setNft(nftValue);
    }
  }, [nftValue]);

  return { nft };
}

export default useOpenSea;
