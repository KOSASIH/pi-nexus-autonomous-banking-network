const VestingDataViewer = ({ data }) => {
  return (
    <div>
      <h2>Vested Tokens: {data.vestedTokens} Pi Nexus Tokens</h2>
      <h2>Estimated Rewards: {data.estimatedRewards} Pi Nexus Tokens</h2>
    </div>
  );
};

export default VestingDataViewer;
