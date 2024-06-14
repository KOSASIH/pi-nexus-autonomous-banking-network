const SwapDataViewer = ({ data }) => {
  return (
    <div>
      <h2>Swap History:</h2>
      <ul>
        {data.swapHistory.map((swap) => (
          <li key={swap.id}>
            {swap.amount} Pi Nexus Tokens -> {swap.targetCurrency}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default SwapDataViewer;
