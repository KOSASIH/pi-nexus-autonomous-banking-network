import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Web3 from 'web3';

const VerifyUser = () => {
  const [username, setUsername] = useState('');
  const [accountAddress, setAccountAddress] = useState('');
  const [password, setPassword] = useState('');
  const [digiCode, setDigiCode] = useState('');
  const [verified, setVerified] = useState(false);
  const [web3, setWeb3] = useState(null);
  const [contract, setContract] = useState(null);

  useEffect(() => {
    const initWeb3 = async () => {
      const web3 = new Web3(
        new Web3.providers.HttpProvider('https://sidra-chain-node.com'),
      );
      setWeb3(web3);
    };
    initWeb3();
  }, []);

  useEffect(() => {
    const initContract = async () => {
      const contract = new web3.eth.Contract(
        [
          {
            constant: true,
            inputs: [],
            name: 'getUserAddress',
            outputs: [{ name: '', type: 'address' }],
            payable: false,
            stateMutability: 'view',
            type: 'function',
          },
          {
            constant: true,
            inputs: [],
            name: 'getSignatureHash',
            outputs: [{ name: '', type: 'string' }],
            payable: false,
            stateMutability: 'view',
            type: 'function',
          },
        ],
        '0x...SidraChainContractAddress...',
      );
      setContract(contract);
    };
    initContract();
  }, [web3]);

  const handleVerify = async () => {
    try {
      const authValidation = await AuthValidation(
        username,
        accountAddress,
        password,
        digiCode,
        web3,
        contract,
      );
      setVerified(authValidation);
    } catch (error) {
      console.error(`Error verifying user: ${error}`);
    }
  };

  return (
    <div>
      <h1>Verify User</h1>
      <form>
        <label>Username:</label>
        <input
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
        <br />
        <label>Account Address:</label>
        <input
          type="text"
          value={accountAddress}
          onChange={(e) => setAccountAddress(e.target.value)}
        />
        <br />
        <label>Password:</label>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <br />
        <label>Digi Code:</label>
        <input
          type="text"
          value={digiCode}
          onChange={(e) => setDigiCode(e.target.value)}
        />
        <br />
        <button onClick={handleVerify}>Verify</button>
      </form>
      {verified ? <p>Verified!</p> : <p>Not verified</p>}
    </div>
  );
};

export default VerifyUser;
