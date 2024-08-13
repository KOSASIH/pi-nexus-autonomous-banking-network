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
import { EdgeProtocol } from '../EdgeProtocol';

const EdgeComponent = ({ edge }) => {
  const [edgeData, setEdgeData] = useState(null);
  const [edgeStatus, setEdgeStatus] = useState(null);
  const [edgeMetrics, setEdgeMetrics] = useState(null);
  const [edgeNeighbors, setEdgeNeighbors] = useState(null);
  const [edgeAIModel, setEdgeAIModel] = useState(null);
  const [edgeQuantumResistant, setEdgeQuantumResistant] = useState(null);
  const [edgeNetworkCartography, setEdgeNetworkCartography] = useState(null);
  const [edgeDecentralizedApps, setEdgeDecentralizedApps] = useState(null);
  const [edgeOpenMainnet, setEdgeOpenMainnet] = useState(null);
  const [edgeProtocol, setEdgeProtocol] = useState(null);

  useEffect(() => {
    const init = async () => {
      const piNetwork = new PiNetwork();
      const edgeData = await piNetwork.getEdgeData(edge);
      setEdgeData(edgeData);
      const edgeStatus = await piNetwork.getEdgeStatus(edge);
      setEdgeStatus(edgeStatus);
      const edgeMetrics = await piNetwork.getEdgeMetrics(edge);
      setEdgeMetrics(edgeMetrics);
      const edgeNeighbors = await piNetwork.getEdgeNeighbors(edge);
      setEdgeNeighbors(edgeNeighbors);
      const aiModel = new AIModel();
      const edgeAIModel = await aiModel.getEdgeAIModel(edge);
      setEdgeAIModel(edgeAIModel);
      const quantumResistant = new QuantumResistant();
      const edgeQuantumResistant = await quantumResistant.getEdgeQuantumResistant(edge);
      setEdgeQuantumResistant(edgeQuantumResistant);
      const networkCartography = new NetworkCartography();
      const edgeNetworkCartography = await networkCartography.getEdgeNetworkCartography(edge);
      setEdgeNetworkCartography(edgeNetworkCartography);
      const decentralizedApps = new DecentralizedApps();
      const edgeDecentralizedApps = await decentralizedApps.getEdgeDecentralizedApps(edge);
      setEdgeDecentralizedApps(edgeDecentralizedApps);
      const openMainnet = new OpenMainnet();
      const edgeOpenMainnet = await openMainnet.getEdgeOpenMainnet(edge);
      setEdgeOpenMainnet(edgeOpenMainnet);
      const edgeProtocol = new EdgeProtocol();
      const edgeProtocolData = await edgeProtocol.getEdgeProtocolData(edge);
      setEdgeProtocol(edgeProtocolData);
    };
    init();
  }, [edge]);

  const handleEdgeClick = async () => {
    const piNetwork = new PiNetwork();
    const edgeData = await piNetwork.getEdgeData(edge);
    setEdgeData(edgeData);
    const atlasMap = new AtlasMap();
    atlasMap.updateAtlasMap(edgeData);
  };

  return (
    <div className="edge-component">
      <h2>{edgeData.name}</h2>
      <p>Status: {edgeStatus}</p>
      <p>Metrics: {edgeMetrics}</p>
      <p>Neighbors: {edgeNeighbors}</p>
      <p>AI Model: {edgeAIModel}</p>
      <p>Quantum Resistant: {edgeQuantumResistant}</p>
      <p>Network Cartography: {edgeNetworkCartography}</p>
      <p>Decentralized Apps: {edgeDecentralizedApps}</p>
      <p>Open Mainnet: {edgeOpenMainnet}</p>
      <p>Edge Protocol: {edgeProtocol}</p>
      <button onClick={handleEdgeClick}>Update Edge Data</button>
    </div>
  );
};

export default EdgeComponent;
