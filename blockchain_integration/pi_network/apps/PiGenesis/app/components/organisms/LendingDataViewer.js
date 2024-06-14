const LendingDataViewer = ({ data }) => {
  return (
    <div>
      <h2>Lent Tokens: {data.lentTokens} Pi Nexus Tokens</h2>
      <h2>Estimated Interest: {data.estimatedInterest} Pi Nexus Tokens</h2>
    </div>
  );
};

export default LendingDataViewer;
