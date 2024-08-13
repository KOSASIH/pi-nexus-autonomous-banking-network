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

const NodeComponent = ({ node }) => {
  const [nodeData, setNodeData] = useState(null);
  const [nodeStatus, setNodeStatus] = useState(null);
  const [nodeMetrics, setNodeMetrics] = useState(null);
  const [nodeNeighbors, setNodeNeighbors] = useState(null);
  const [nodeAIModel, setNodeAIModel] = useState(null);
  const [nodeQuantumResistant, setNodeQuantumResistant] = useState(null);
  const [nodeNetworkCartography, setNodeNetworkCartography] = useState(null);
  const [nodeDecentralizedApps, setNodeDecentralizedApps] = useState(null);
  const [nodeOpenMainnet, setNodeOpenMainnet] = useState(null);

  useEffect(() => {
    const init = async () => {
      const piNetwork = new PiNetwork();
      const nodeData = await piNetwork.getNodeData(node);
      setNodeData(nodeData);
      const nodeStatus = await piNetwork.getNodeStatus(node);
      setNodeStatus(nodeStatus);
      const nodeMetrics = await piNetwork.getNodeMetrics(node);
      setNodeMetrics(nodeMetrics);
      const nodeNeighbors = await piNetwork.getNodeNeighbors(node);
      setNodeNeighbors(nodeNeighbors);
      const aiModel = new AIModel();
      const nodeAIModel = await aiModel.getNodeAIModel(node);
      setNodeAIModel(nodeAIModel);
      const quantumResistant = new QuantumResistant();
      const nodeQuantumResistant = await quantumResistant.getNodeQuantumResistant(node);
      setNodeQuantumResistant(nodeQuantumResistant);
      const networkCartography = new NetworkCartography();
      const nodeNetworkCartography = await networkCartography.getNodeNetworkCartography(node);
      setNodeNetworkCartography(nodeNetworkCartography);
      const decentralizedApps = new DecentralizedApps();
      const nodeDecentralizedApps = await decentralizedApps.getNodeDecentralizedApps(node);
      setNodeDecentralizedApps(nodeDecentralizedApps);
      const openMainnet = new OpenMainnet();
      const nodeOpenMainnet = await openMainnet.getNodeOpenMainnet(node);
      setNodeOpenMainnet(nodeOpenMainnet);
    };
    init();
  }, [node]);

  const handleNodeClick = async () => {
    const piNetwork = new PiNetwork();
    const nodeData = await piNetwork.getNodeData(node);
    setNodeData(nodeData);
    const atlasMap = new AtlasMap();
    atlasMap.updateAtlasMap(nodeData);
  };

  return (
    <div className="node-component">
      <h2>{nodeData.name}</h2>
      <p>Status: {nodeStatus}</p>
      <p>Metrics: {nodeMetrics}</p>
      <p>Neighbors: {nodeNeighbors}</p>
      <p>AI Model: {nodeAIModel}</p>
      <p>Quantum Resistant: {nodeQuantumResistant}</p>
      <p>Network Cartography: {nodeNetworkCartography}</p>
      <p>Decentralized Apps: {nodeDecentralizedApps}</p>
      <p>Open Mainnet: {nodeOpenMainnet}</p>
      <button onClick={handleNodeClick}>Update Node Data</button>
    </div>
  );
};

export default NodeComponent;
