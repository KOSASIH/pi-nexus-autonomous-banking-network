const StakingDataViewer = ({ data }) => {
  return (
    <div>
      <h2>Staked Tokens: {data.stakedTokens} Pi Nexus Tokens</h2>
      <h2>Estimated Rewards: {data.estimatedRewards} Pi Nexus Tokens</h2>
    </div>
  );
};

export default StakingDataViewer;
