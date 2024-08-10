import React, { useState, useEffect, useMemo } from 'react';
import { useAuth } from '../hooks/useAuth';
import { useNodeData } from '../hooks/useNodeData';
import { useNodeSelection } from '../hooks/useNodeSelection';
import { useNodeIncentivization } from '../hooks/useNodeIncentivization';
import { useNodeReputation } from '../hooks/useNodeReputation';
import { NodeList } from '../components/NodeList';
import { NodeCard } from '../components/NodeCard';
import { NodeFilters } from '../components/NodeFilters';
import { NodeSort } from '../components/NodeSort';
import { NodePagination } from '../components/NodePagination';
import { NodeActions } from '../components/NodeActions';
import { NodeReputationChart } from '../components/NodeReputationChart';
import { NodeIncentivizationChart } from '../components/NodeIncentivizationChart';
import { DashboardHeader } from '../components/DashboardHeader';
import { DashboardSidebar } from '../components/DashboardSidebar';

const Dashboard = () => {
  const { user, logout } = useAuth();
  const [nodes, setNodes] = useState([]);
  const [selectedNodes, setSelectedNodes] = useState([]);
  const [incentivizationData, setIncentivizationData] = useState({});
  const [reputationData, setReputationData] = useState({});

  const { data, error, isLoading } = useNodeData();
  const { nodeSelection, setNodeSelection } = useNodeSelection();
  const { nodeIncentivization, setNodeIncentivization } = useNodeIncentivization();
  const { nodeReputation, setNodeReputation } = useNodeReputation();

  useEffect(() => {
    if (data) {
      setNodes(data.nodes);
      setIncentivizationData(data.incentivization);
      setReputationData(data.reputation);
    }
  }, [data]);

  const handleNodeSelect = (nodeId) => {
    setSelectedNodes((prev) => [...prev, nodeId]);
  };

  const handleNodeDeselect = (nodeId) => {
    setSelectedNodes((prev) => prev.filter((id) => id !== nodeId));
  };

  const handleLogout = () => {
    logout();
  };

  return (
    <div className="dashboard">
      <DashboardHeader user={user} onLogout={handleLogout} />
      <DashboardSidebar />
      <div className="dashboard-content">
        <NodeFilters />
        <NodeSort />
        <NodeList
          nodes={nodes}
          selectedNodes={selectedNodes}
          onSelect={handleNodeSelect}
          onDeselect={handleNodeDeselect}
        />
        <NodePagination />
        <NodeCard nodes={selectedNodes} />
        <NodeActions nodes={selectedNodes} />
        <NodeReputationChart data={reputationData} />
        <NodeIncentivizationChart data={incentivizationData} />
      </div>
    </div>
  );
};

export default Dashboard;
