const BurnDataViewer = ({ data }) => {
  return (
    <div>
      <h2>Burned Tokens: {data.burnedTokens} Pi Nexus Tokens</h2>
      <h2>Estimated Rewards: {data.estimatedRewards} Pi Nexus Tokens</h2>
    </div>
  );
};

export default BurnDataViewer;
