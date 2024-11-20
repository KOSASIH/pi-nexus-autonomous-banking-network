import web3 from './web3';

export const getContractInstance = (abi, address) => {
    return new web3.eth.Contract(abi, address);
};

```javascript
export const callContractMethod = async (contract, method, args) => {
    return await contract.methods[method](...args).call();
};

export const sendTransaction = async (contract, method, args, from) => {
    return await contract.methods[method](...args).send({ from });
};
