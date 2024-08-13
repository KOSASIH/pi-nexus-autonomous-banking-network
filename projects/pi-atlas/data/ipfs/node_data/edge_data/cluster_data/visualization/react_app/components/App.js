import React, { useState, useEffect } from 'react';
import { BrowserRouter, Route, Switch } from 'react-router-dom';
import { Web3Provider } from '@ethersproject/providers';
import { useWeb3React } from '@web3-react/core';
import { PiNetwork } from './PiNetwork';
import { AtlasMap } from './AtlasMap';
import { AIModel } from './AIModel';
import { QuantumResistant } from './QuantumResistant';
import { NetworkCartography } from './NetworkCartography';
import { DecentralizedApps } from './DecentralizedApps';
import { OpenMainnet } from './OpenMainnet';

const App = () => {
  const [account, setAccount] = useState(null);
  const [provider, setProvider] = useState(null);
  const [network, setNetwork] = useState(null);
  const [atlasMap, setAtlasMap] = useState(null);
  const [aiModel, setAiModel] = useState(null);
  const [quantumResistant, setQuantumResistant] = useState(null);
  const [networkCartography, setNetworkCartography] = useState(null);
  const [decentralizedApps, setDecentralizedApps] = useState(null);
  const [openMainnet, setOpenMainnet] = useState(null);

  useEffect(() => {
    const init = async () => {
      const provider = new Web3Provider(window.ethereum);
      const account = await provider.listAccounts();
      setAccount(account[0]);
      setProvider(provider);
      const piNetwork = new PiNetwork(provider);
      setNetwork(piNetwork);
      const atlasMap = new AtlasMap(piNetwork);
      setAtlasMap(atlasMap);
      const aiModel = new AIModel(atlasMap);
      setAiModel(aiModel);
      const quantumResistant = new QuantumResistant(aiModel);
      setQuantumResistant(quantumResistant);
      const networkCartography = new NetworkCartography(quantumResistant);
      setNetworkCartography(networkCartography);
      const decentralizedApps = new DecentralizedApps(networkCartography);
      setDecentralizedApps(decentralizedApps);
      const openMainnet = new OpenMainnet(decentralizedApps);
      setOpenMainnet(openMainnet);
    };
    init();
  }, []);

  return (
    <BrowserRouter>
      <Switch>
        <Route path="/" exact component={AtlasMap} />
        <Route path="/ai-model" component={AIModel} />
        <Route path="/quantum-resistant" component={QuantumResistant} />
        <Route path="/network-cartography" component={NetworkCartography} />
        <Route path="/decentralized-apps" component={DecentralizedApps} />
        <Route path="/open-mainnet" component={OpenMainnet} />
      </Switch>
    </BrowserRouter>
  );
};

export default App;
