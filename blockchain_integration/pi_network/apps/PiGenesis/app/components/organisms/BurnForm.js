import { useState } from 'eact';

const BurnForm = ({ onSubmit }) => {
  const [amount, setAmount] = useState(0);

  const handleSubmit = (event) => {
    event.preventDefault();
    onSubmit(amount);
  };

  return (
    <form onSubmit={handleSubmit}>
      <label htmlFor="burn-amount">Burn Amount:</label>
      <input
        type="number"
        id="burn-amount"
        value={amount}
        onChange={(event) => setAmount(event.target.value)}
      />
      <button type="submit">Burn Tokens</button>
    </form>
  );
};

export default BurnForm;
