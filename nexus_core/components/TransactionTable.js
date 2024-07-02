import React from 'eact';

const TransactionTable = () => {
  const transactions = [
    { id: 1, date: '2024-07-01', type: 'Deposit', amount: 100 },
    { id: 2, date: '2024-07-02', type: 'Withdrawal', amount: 50 },
    { id: 3, date: '2024-07-03', type: 'Transfer', amount: 200 },
  ];

  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Date</th>
          <th>Type</th>
          <th>Amount</th>
        </tr>
      </thead>
      <tbody>
        {transactions.map((transaction) => (
          <tr key={transaction.id}>
            <td>{transaction.id}</td>
            <td>{transaction.date}</td>
            <td>{transaction.type}</td>
            <td>{transaction.amount}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

export default TransactionTable;
