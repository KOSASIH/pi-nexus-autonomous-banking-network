// NodePerformanceContainer.js

import React, { useState, useEffect } from 'react';
import { useRecoilState } from 'recoil';
import { nodePerformanceDataState } from '../atoms/nodePerformanceData';
import NodePerformanceChart from '../components/NodePerformanceChart';
import NodeOptimizerTrainer from '../components/NodeOptimizerTrainer';

const NodePerformanceContainer = () => {
  const [nodePerformanceData, setNodePerformanceData] = useRecoilState(nodePerformanceDataState);
  const [optimizerTrained, setOptimizerTrained] = useState(false);

  useEffect(() => {
    const fetchNodePerformanceData = async () => {
      const response = await fetch('/api/node-performance-data');
      const data = await response.json();
      setNodePerformanceData(data);
    };
    fetchNodePerformanceData();
  }, []);

  const handleOptimizerTraining = async () => {
    const optimizer = new NodeOptimizerTrainer(nodePerformanceData);
    await optimizer.train();
    setOptimizerTrained(true);
  };

  return (
    <div>
      <h2>Node Performance Chart</h2>
      <NodePerformanceChart />
      {optimizerTrained ? (
        <p>Optimizer trained successfully!</p>
      ) : (
        <button onClick={handleOptimizerTraining}>Train Node Optimizer</button>
      )}
    </div>
  );
};

export default NodePerformanceContainer;
