import React, { useState, useEffect } from 'react';
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

const ClusterComponent = ({ cluster }) => {
  const [clusterData, setClusterData] = useState(null);
  const [clusterStatus, setClusterStatus] = useState(null);
  const [clusterMetrics, setClusterMetrics] = useState(null);
  const [clusterNodes, setClusterNodes] = useState(null);
  const [clusterEdges, setClusterEdges] = useState(null);
  const [clusterAIModel, setClusterAIModel] = useState(null);
  const [clusterQuantumResistant, setClusterQuantumResistant] = useState(null);
  const [clusterNetworkCartography, setClusterNetworkCartography] = useState(null);
  const [clusterDecentralizedApps, setClusterDecentralizedApps] = useState(null);
  const [clusterOpenMainnet, setClusterOpenMainnet] = useState(null);
  const [clusterProtocol, setClusterProtocol] = useState(null);

  useEffect(() => {
    const init = async () => {
      const piNetwork = new PiNetwork();
      const clusterData = await piNetwork.getClusterData(cluster);
      setClusterData(clusterData);
      const clusterStatus = await piNetwork.getClusterStatus(cluster);
      setClusterStatus(clusterStatus);
      const clusterMetrics = await piNetwork.getClusterMetrics(cluster);
      setClusterMetrics(clusterMetrics);
      const clusterNodes = await piNetwork.getClusterNodes(cluster);
      setClusterNodes(clusterNodes);
      const clusterEdges = await piNetwork.getClusterEdges(cluster);
      setClusterEdges(clusterEdges);
      const aiModel = new AIModel();
      const clusterAIModel = await aiModel.getClusterAIModel(cluster);
      setClusterAIModel(clusterAIModel);
      const quantumResistant = new QuantumResistant();
      const clusterQuantumResistant = await quantumResistant.getClusterQuantumResistant(cluster);
      setClusterQuantumResistant(clusterQuantumResistant);
      const networkCartography = new NetworkCartography();
      const clusterNetworkCartography = await networkCartography.getClusterNetworkCartography(cluster);
      setClusterNetworkCartography(clusterNetworkCartography);
      const decentralizedApps = new DecentralizedApps();
      const clusterDecentralizedApps = await decentralizedApps.getClusterDecentralizedApps(cluster);
      setClusterDecentralizedApps(clusterDecentralizedApps);
      const openMainnet = new OpenMainnet();
      const clusterOpenMainnet = await openMainnet.getClusterOpenMainnet(cluster);
      setClusterOpenMainnet(clusterOpenMainnet);
      const clusterProtocol = new ClusterProtocol();
      const clusterProtocolData = await clusterProtocol.getClusterProtocolData(cluster);
      setClusterProtocol(clusterProtocolData);
    };
    init();
  }, [cluster]);

  const handleClusterClick = async () => {
    const piNetwork = new PiNetwork();
    const clusterData = await piNetwork.getClusterData(cluster);
    setClusterData(clusterData);
    const atlasMap = new AtlasMap();
    atlasMap.updateAtlasMap(clusterData);
  };

  return (
    <div className="cluster-component">
      <h2>{clusterData.name}</h2>
      <p>Status: {clusterStatus}</p>
      <p>Metrics: {clusterMetrics}</p>
      <p>Nodes: {clusterNodes}</p>
      <p>Edges: {clusterEdges}</p>
      <p>AI Model: {clusterAIModel}</p>
      <p>Quantum Resistant: {clusterQuantumResistant}</p>
      <p>Network Cartography: {clusterNetworkCartography}</p>
      <p>Decentralized Apps: {clusterDecentralizedApps}</p>
      <p>Open Mainnet: {clusterOpenMainnet}</p>
      <p>Cluster Protocol: {clusterProtocol}</p>
      <button onClick={handleClusterClick}>Update Cluster Data</button>
    </div>
  );
};

export default ClusterComponent;
