import React, { useState, useEffect, useContext } from 'react';
import { useWeb3React } from '@web3-react/core';
import { ethers } from 'ethers';
import { Cluster } from '../Cluster';
import { ClusterService } from '../ClusterService';
import { ClusterMetrics } from '../ClusterMetrics';
import { ClusterSecurity } from '../ClusterSecurity';
import { ClusterNetworking } from '../ClusterNetworking';
import { ClusterStorage } from '../ClusterStorage';
import { ClusterAI } from '../ClusterAI';
import { ClusterQuantum } from '../ClusterQuantum';
import { AppContext } from '../AppContext';
import { EdgeContainer } from './EdgeContainer';
import { NodeContainer } from './NodeContainer';

const ClusterContainer = ({ cluster }) => {
  const [clusterState, setClusterState] = useState({
    status: 'offline',
    metrics: {},
    security: {},
    networking: {},
    storage: {},
    ai: {},
    quantum: {},
    edges: [],
    nodes: [],
  });

  const { account, library } = useWeb3React();
  const { appState, handleAppStateChange } = useContext(AppContext);

  useEffect(() => {
    const init = async () => {
      const clusterService = new ClusterService(cluster);
      const clusterMetrics = await clusterService.getMetrics();
      const clusterSecurity = await clusterService.getSecurity();
      const clusterNetworking = await clusterService.getNetworking();
      const clusterStorage = await clusterService.getStorage();
      const clusterAI = await clusterService.getAI();
      const clusterQuantum = await clusterService.getQuantum();
      const edges = await clusterService.getEdges();
      const nodes = await clusterService.getNodes();
      setClusterState({
        status: 'online',
        metrics: clusterMetrics,
        security: clusterSecurity,
        networking: clusterNetworking,
        storage: clusterStorage,
        ai: clusterAI,
        quantum: clusterQuantum,
        edges,
        nodes,
      });
    };
    init();
  }, [cluster]);

  useEffect(() => {
    const updateClusterState = async () => {
      const clusterService = new ClusterService(cluster);
      const clusterMetrics = await clusterService.getMetrics();
      const clusterSecurity = await clusterService.getSecurity();
      const clusterNetworking = await clusterService.getNetworking();
      const clusterStorage = await clusterService.getStorage();
      const clusterAI = await clusterService.getAI();
      const clusterQuantum = await clusterService.getQuantum();
      const edges = await clusterService.getEdges();
      const nodes = await clusterService.getNodes();
      setClusterState({
        metrics: clusterMetrics,
        security: clusterSecurity,
        networking: clusterNetworking,
        storage: clusterStorage,
        ai: clusterAI,
        quantum: clusterQuantum,
        edges,
        nodes,
      });
    };
    updateClusterState();
  }, [clusterState]);

  const handleClusterStatusChange = async (newStatus) => {
    setClusterState({ status: newStatus });
    const clusterService = new ClusterService(cluster);
    await clusterService.updateStatus(newStatus);
  };

  const handleClusterMetricsChange = async (newMetrics) => {
    setClusterState({ metrics: newMetrics });
    const clusterService = new ClusterService(cluster);
    await clusterService.updateMetrics(newMetrics);
  };

  const handleClusterSecurityChange = async (newSecurity) => {
    setClusterState({ security: newSecurity });
    const clusterService = new ClusterService(cluster);
    await clusterService.updateSecurity(newSecurity);
  };

  const handleClusterNetworkingChange = async (newNetworking) => {
    setClusterState({ networking: newNetworking });
    const clusterService = new ClusterService(cluster);
    await clusterService.updateNetworking(newNetworking);
  };

  const handleClusterStorageChange = async (newStorage) => {
    setClusterState({ storage: newStorage });
    const clusterService = new ClusterService(cluster);
    await clusterService.updateStorage(newStorage);
  };

  const handleClusterAIChange = async (newAI) => {
    setClusterState({ ai: newAI });
    const clusterService = new ClusterService(cluster);
    await clusterService.updateAI(newAI);
  };

  const handleClusterQuantumChange = async (newQuantum) => {
    setClusterState({ quantum: newQuantum });
    const clusterService = new ClusterService(cluster);
    await clusterService.updateQuantum(newQuantum);
  };

  const handleEdgeAdd = async (newEdge) => {
    const clusterService = new ClusterService(cluster);
    await clusterService.addEdge(newEdge);
    setClusterState({ edges: [...clusterState.edges, newEdge] });
  };

  const handleEdgeRemove = async (edgeId) => {
    const clusterService = new ClusterService(cluster);
    await clusterService.removeEdge(edgeId);
    setClusterState({ edges: clusterState.edges.filter((edge) => edge.id !== edgeId) });
  };
