const WalletDataViewer = ({ data }) => {
  return (
    <div>
      <h2>Balance: {data.balance} Pi Nexus Tokens</h2>
      <h2>Transaction History:</h2>
      <ul>
        {data.transactions.map((transaction) => (
          <li key={transaction.id}>
            {transaction.type} - {transaction.amount} Pi Nexus Tokens
          </li>
        ))}
      </ul>
    </div>
  );
};

export default WalletDataViewer;
