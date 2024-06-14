import { useState, useEffect } from 'eact';
import { useBlockchain } from '@pi-nexus/blockchain-react';

const AnalyticsDataViewer = ({ data }) => {
  const [insights, setInsights] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const { blockchain } = useBlockchain();

  useEffect(() => {
    const fetchInsights = async () => {
      const insightsData = await blockchain.getInsightsData(data.insightsId);
      setInsights(insightsData);
    };

    const fetchPredictions = async () => {
      const predictionsData = await blockchain.getPredictionsData(data.predictionsId);
      setPredictions(predictionsData);
    };

   fetchInsights();
    fetchPredictions();
  }, [data, blockchain]);

  return (
    <div>
      <h2>Insights</h2>
      {insights && (
        <InsightsViewer data={insights} />
      )}
      <h2>Predictions</h2>
      {predictions && (
        <PredictionsViewer data={predictions} />
      )}
    </div>
  );
};

export default AnalyticsDataViewer;
