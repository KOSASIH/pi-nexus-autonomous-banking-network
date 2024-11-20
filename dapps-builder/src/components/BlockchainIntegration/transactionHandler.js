import web3 from './web3';

export const createTransaction = async (to, value, from) => {
    const transaction = {
        to,
        value: web3.utils.toWei(value.toString(), 'ether'),
        gas: 2000000,
        from
    };
    return await web3.eth.sendTransaction(transaction);
};

export const getTransactionReceipt = async (txHash) => {
    return await web3.eth.getTransactionReceipt(txHash);
};
