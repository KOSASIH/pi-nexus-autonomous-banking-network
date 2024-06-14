import { useState } from 'eact';

const LendForm = ({ onSubmit }) => {
  const [amount, setAmount] = useState(0);
  const [interestRate, setInterestRate] = useState('');

  const handleSubmit = (event) => {
    event.preventDefault();
    onSubmit(amount, interestRate);
  };

  return (
    <form onSubmit={handleSubmit}>
      <label htmlFor="lend-amount">Lend Amount:</label>
      <input
        type="number"
        id="lend-amount"
        value={amount}
        onChange={(event) => setAmount(event.target.value)}
      />
      <label htmlFor="interest-rate">Interest Rate:</label>
      <input
        type="text"
        id="interest-rate"
        value={interestRate}
        onChange={(event) => setInterestRate(event.target.value)}
      />
      <button type="submit">Lend Tokens</button>
    </form>
  );
};

export default LendForm;
