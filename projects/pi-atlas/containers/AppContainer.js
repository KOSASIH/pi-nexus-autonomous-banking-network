import React, { useState, useEffect, useContext } from 'react';
import { useWeb3React } from '@web3-react/core';
import { ethers } from 'ethers';
import { PiNetwork } from '../PiNetwork';
import { AtlasMap } from '../AtlasMap';
import { AIModel } from '../AIModel';
import { QuantumResistant } from '../QuantumResistant';
import { NetworkCartography } from '../NetworkCartography';
import { DecentralizedApps } from '../DecentralizedApps';
import { OpenMainnet } from '../OpenMainnet';
import { ClusterProtocol } from '../ClusterProtocol';
import { DistributedStorage } from '../DistributedStorage';
import { AutonomousAgents } from '../AutonomousAgents';
import { SwarmIntelligence } from '../SwarmIntelligence';
import { AppContext } from '../AppContext';

const AppContainer = () => {
  const [clusters, setClusters] = useState([]);
  const [selectedCluster, setSelectedCluster] = useState(null);
  const [appState, setAppState] = useState({
    networkStatus: 'offline',
    aiModelStatus: 'idle',
    quantumResistantStatus: 'inactive',
    networkCartographyStatus: 'loading',
    decentralizedAppsStatus: 'disabled',
    openMainnetStatus: 'closed',
    clusterProtocolStatus: 'inactive',
    distributedStorageStatus: 'offline',
    autonomousAgentsStatus: 'idle',
    swarmIntelligenceStatus: 'inactive',
  });

  const { account, library } = useWeb3React();

  useEffect(() => {
    const init = async () => {
      const piNetwork = new PiNetwork();
      const clusters = await piNetwork.getClusters();
      setClusters(clusters);
      const appState = await piNetwork.getAppState();
      setAppState(appState);
    };
    init();
  }, []);

  useEffect(() => {
    const updateAppState = async () => {
      const piNetwork = new PiNetwork();
      const appState = await piNetwork.getAppState();
      setAppState(appState);
    };
    updateAppState();
  }, [selectedCluster]);

  const handleClusterSelect = async (cluster) => {
    setSelectedCluster(cluster);
    const piNetwork = new PiNetwork();
    const appState = await piNetwork.getAppState(cluster);
    setAppState(appState);
  };

  const handleAppStateChange = async (newAppState) => {
    setAppState(newAppState);
    const piNetwork = new PiNetwork();
    await piNetwork.updateAppState(newAppState);
  };

  return (
    <AppContext.Provider value={{ appState, handleAppStateChange }}>
      <div className="app-container">
        <h1>Pi Atlas</h1>
        <p>Network Status: {appState.networkStatus}</p>
        <p>AI Model Status: {appState.aiModelStatus}</p>
        <p>Quantum Resistant Status: {appState.quantumResistantStatus}</p>
        <p>Network Cartography Status: {appState.networkCartographyStatus}</p>
        <p>Decentralized Apps Status: {appState.decentralizedAppsStatus}</p>
        <p>Open Mainnet Status: {appState.openMainnetStatus}</p>
        <p>Cluster Protocol Status: {appState.clusterProtocolStatus}</p>
        <p>Distributed Storage Status: {appState.distributedStorageStatus}</p>
        <p>Autonomous Agents Status: {appState.autonomousAgentsStatus}</p>
        <p>Swarm Intelligence Status: {appState.swarmIntelligenceStatus}</p>
        <ClusterList clusters={clusters} onSelect={handleClusterSelect} />
        {selectedCluster && (
          <ClusterComponent cluster={selectedCluster} />
        )}
      </div>
    </AppContext.Provider>
  );
};

export default AppContainer;
