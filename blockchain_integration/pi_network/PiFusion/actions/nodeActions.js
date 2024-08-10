import React, { useState, useEffect } from 'react';
import { useAuth } from '../../hooks/useAuth';
import { useNodeData } from '../../hooks/useNodeData';
import { useNodeSelection } from '../../hooks/useNodeSelection';
import { useNodeIncentivization } from '../../hooks/useNodeIncentivization';
import { useNodeReputation } from '../../hooks/useNodeReputation';
import { NodeActionButton } from './NodeActionButton';
import { NodeActionMenu } from './NodeActionMenu';
import { NodeActionModal } from './NodeActionModal';

const NodeActions = () => {
  const { user } = useAuth();
  const { nodes } = useNodeData();
  const { selectedNodes } = useNodeSelection();
  const { incentivizationData } = useNodeIncentivization();
  const { reputationData } = useNodeReputation();
  const [actionModalOpen, setActionModalOpen] = useState(false);
  const [actionType, setActionType] = useState('');
  const [actionNode, setActionNode] = useState(null);

  useEffect(() => {
    if (selectedNodes.length === 0) {
      setActionModalOpen(false);
    }
  }, [selectedNodes]);

  const handleActionButtonClick = (actionType, node) => {
    setActionModalOpen(true);
    setActionType(actionType);
    setActionNode(node);
  };

  const handleActionModalClose = () => {
    setActionModalOpen(false);
  };

  const handleActionSubmit = (actionData) => {
    // Call API to perform action on selected nodes
    console.log('Performing action:', actionData);
    setActionModalOpen(false);
  };

  return (
    <div className="node-actions">
      {selectedNodes.map((node) => (
        <NodeActionButton
          key={node.id}
          node={node}
          incentivizationData={incentivizationData}
          reputationData={reputationData}
          onClick={(actionType) => handleActionButtonClick(actionType, node)}
        />
      ))}
      {actionModalOpen && (
        <NodeActionModal
          isOpen={actionModalOpen}
          onClose={handleActionModalClose}
          actionType={actionType}
          node={actionNode}
          onSubmit={handleActionSubmit}
        />
      )}
      <NodeActionMenu />
    </div>
  );
};

export default NodeActions;
