import { useState, useEffect } from 'eact';
import { useBlockchain } from '@pi-nexus/blockchain-react';
import { useMachineLearning } from '@pi-nexus/machine-learning-react';

const RealtimeAnalytics = () => {
  const [analyticsData, setAnalyticsData] = useState(null);
  const { blockchain } = useBlockchain();
  const { machineLearning } = useMachineLearning();

  useEffect(() => {
    const fetchAnalyticsData = async () => {
      const data = await blockchain.getAnalyticsData();
      setAnalyticsData(data);
    };

    fetchAnalyticsData();
  }, [blockchain]);

  const handlePredict = async () => {
    const prediction = await machineLearning.predict(analyticsData);
    console.log(prediction);
  };

  return (
    <div>
      <h1>Real-time Analytics</h1>
      {analyticsData && (
        <AnalyticsDataViewer data={analyticsData} />
      )}
      <button onClick={handlePredict}>Get Predictions</button>
    </div>
  );
};

export default RealtimeAnalytics;
