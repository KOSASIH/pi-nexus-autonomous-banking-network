import { useState } from 'eact';

constStakeForm = ({ onSubmit }) => {
  const [amount, setAmount] = useState(0);

  const handleSubmit = (event) => {
    event.preventDefault();
    onSubmit(amount);
  };

  return (
    <form onSubmit={handleSubmit}>
      <label htmlFor="stake-amount">Stake Amount:</label>
      <input
        type="number"
        id="stake-amount"
        value={amount}
        onChange={(event) => setAmount(event.target.value)}
      />
      <button type="submit">Stake Tokens</button>
    </form>
  );
};

export default StakeForm;
