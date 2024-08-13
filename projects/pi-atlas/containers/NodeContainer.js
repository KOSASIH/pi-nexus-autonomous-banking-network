import React, { useState, useEffect, useContext } from 'react';
import { useWeb3React } from '@web3-react/core';
import { ethers } from 'ethers';
import { Node } from '../Node';
import { NodeService } from '../NodeService';
import { NodeMetrics } from '../NodeMetrics';
import { NodeSecurity } from '../NodeSecurity';
import { NodeNetworking } from '../NodeNetworking';
import { NodeStorage } from '../NodeStorage';
import { NodeAI } from '../NodeAI';
import { NodeQuantum } from '../NodeQuantum';
import { AppContext } from '../AppContext';

const NodeContainer = ({ node }) => {
  const [nodeState, setNodeState] = useState({
    status: 'offline',
    metrics: {},
    security: {},
    networking: {},
    storage: {},
    ai: {},
    quantum: {},
  });

  const { account, library } = useWeb3React();
  const { appState, handleAppStateChange } = useContext(AppContext);

  useEffect(() => {
    const init = async () => {
      const nodeService = new NodeService(node);
      const nodeMetrics = await nodeService.getMetrics();
      const nodeSecurity = await nodeService.getSecurity();
      const nodeNetworking = await nodeService.getNetworking();
      const nodeStorage = await nodeService.getStorage();
      const nodeAI = await nodeService.getAI();
      const nodeQuantum = await nodeService.getQuantum();
      setNodeState({
        status: 'online',
        metrics: nodeMetrics,
        security: nodeSecurity,
        networking: nodeNetworking,
        storage: nodeStorage,
        ai: nodeAI,
        quantum: nodeQuantum,
      });
    };
    init();
  }, [node]);

  useEffect(() => {
    const updateNodeState = async () => {
      const nodeService = new NodeService(node);
      const nodeMetrics = await nodeService.getMetrics();
      const nodeSecurity = await nodeService.getSecurity();
      const nodeNetworking = await nodeService.getNetworking();
      const nodeStorage = await nodeService.getStorage();
      const nodeAI = await nodeService.getAI();
      const nodeQuantum = await nodeService.getQuantum();
      setNodeState({
        metrics: nodeMetrics,
        security: nodeSecurity,
        networking: nodeNetworking,
        storage: nodeStorage,
        ai: nodeAI,
        quantum: nodeQuantum,
      });
    };
    updateNodeState();
  }, [nodeState]);

  const handleNodeStatusChange = async (newStatus) => {
    setNodeState({ status: newStatus });
    const nodeService = new NodeService(node);
    await nodeService.updateStatus(newStatus);
  };

  const handleNodeMetricsChange = async (newMetrics) => {
    setNodeState({ metrics: newMetrics });
    const nodeService = new NodeService(node);
    await nodeService.updateMetrics(newMetrics);
  };

  const handleNodeSecurityChange = async (newSecurity) => {
    setNodeState({ security: newSecurity });
    const nodeService = new NodeService(node);
    await nodeService.updateSecurity(newSecurity);
  };

  const handleNodeNetworkingChange = async (newNetworking) => {
    setNodeState({ networking: newNetworking });
    const nodeService = new NodeService(node);
    await nodeService.updateNetworking(newNetworking);
  };

  const handleNodeStorageChange = async (newStorage) => {
    setNodeState({ storage: newStorage });
    const nodeService = new NodeService(node);
    await nodeService.updateStorage(newStorage);
  };

  const handleNodeAIChange = async (newAI) => {
    setNodeState({ ai: newAI });
    const nodeService = new NodeService(node);
    await nodeService.updateAI(newAI);
  };

  const handleNodeQuantumChange = async (newQuantum) => {
    setNodeState({ quantum: newQuantum });
    const nodeService = new NodeService(node);
    await nodeService.updateQuantum(newQuantum);
  };

  return (
    <div className="node-container">
      <h2>Node {node.id}</h2>
      <p>Status: {nodeState.status}</p>
      <p>Metrics: {JSON.stringify(nodeState.metrics)}</p>
      <p>Security: {JSON.stringify(nodeState.security)}</p>
      <p>Networking: {JSON.stringify(nodeState.networking)}</p>
      <p>Storage: {JSON.stringify(nodeState.storage)}</p>
      <p>AI: {JSON.stringify(nodeState.ai)}</p>
      <p>Quantum: {JSON.stringify(nodeState.quantum)}</p>
      <button onClick={() => handleNodeStatusChange('online')}>Toggle Online</button>
      <button onClick={() => handleNodeStatusChange('offline')}>Toggle Offline</button>
      <button onClick={() => handleNodeMetricsChange({ cpu: 50, memory:
