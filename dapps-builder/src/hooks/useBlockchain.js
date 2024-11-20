import { useState, useEffect } from 'react';
import BlockchainService from '../services/BlockchainService';

const useBlockchain = (abi, contractAddress) => {
    const [accounts, setAccounts] = useState([]);
    const [networkId, setNetworkId] = useState(null);
    const [error, setError] = useState(null);
    const [contract, setContract] = useState(null);

    useEffect(() => {
        const initBlockchain = async () => {
            try {
                const accounts = await BlockchainService.getAccounts();
                const networkId = await BlockchainService.getNetworkId();
                setAccounts(accounts);
                setNetworkId(networkId);
                setContract(new BlockchainService(abi, contractAddress));
            } catch (err) {
                setError(err.message);
            }
        };

        initBlockchain();
    }, [abi, contractAddress]);

    const callMethod = async (method, args) => {
        if (!contract) {
            throw new Error('Contract not initialized.');
        }
        return await contract.callMethod(method, args);
    };

    const sendMethod = async (method, args, from) => {
        if (!contract) {
            throw new Error('Contract not initialized.');
        }
        return await contract.sendMethod(method, args, from);
    };

    return {
        accounts,
        networkId,
        error,
        callMethod,
        sendMethod,
    };
};

export default useBlockchain;
