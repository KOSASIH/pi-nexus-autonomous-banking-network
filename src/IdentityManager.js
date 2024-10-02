import React, { useState, useEffect } from "eact";
import Web3 from "web3";
import { contract } from "truffle-contract";

const IdentityManager = () => {
  const [contractInstance, setContractInstance] = useState(null);
  const [numIdentities, setNumIdentities] = useState(0);
  const [identities, setIdentities] = useState([]);
  const [newIdentity, setNewIdentity] = useState("");

  useEffect(() => {
    const initContract = async () => {
      const web3 = new Web3(
        new Web3.providers.HttpProvider("http://localhost:7545"),
      );
      const contractInstance = await contract("PiNexusIdentityManager").at(
        "0x...ContractAddress...",
      );
      setContractInstance(contractInstance);
    };
    initContract();
  }, []);

  const handleAddIdentity = async () => {
    if (!newIdentity) return;
    await contractInstance.addIdentity(newIdentity);
    setNewIdentity("");
    updateIdentities();
  };

  const handleRemoveIdentity = async (index) => {
    await contractInstance.removeIdentity(index);
    updateIdentities();
  };

  const updateIdentities = async () => {
    const numIdentities = await contractInstance.getNumIdentities();
    setNumIdentities(numIdentities);
    const identities = [];
    for (let i = 0; i < numIdentities; i++) {
      const identity = await contractInstance.getIdentity(i);
      identities.push(identity);
    }
    setIdentities(identities);
  };

  return (
    <div>
      <h1>PiNexus Identity Manager</h1>
      <p>Number of identities: {numIdentities}</p>
      <ul>
        {identities.map((identity, index) => (
          <li key={index}>
            {identity}{" "}
            <button onClick={() => handleRemoveIdentity(index)}>Remove</button>
          </li>
        ))}
      </ul>
      <input
        type="text"
        value={newIdentity}
        onChange={(e) => setNewIdentity(e.target.value)}
        placeholder="Enter new identity"
      />
      <button onClick={handleAddIdentity}>Add Identity</button>
    </div>
  );
};

export default IdentityManager;
