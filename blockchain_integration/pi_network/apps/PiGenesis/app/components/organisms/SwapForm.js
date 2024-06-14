import { useState } from 'eact';

const SwapForm = ({ onSubmit }) => {
  const [amount, setAmount] = useState(0);
  const [targetCurrency, setTargetCurrency] = useState('');

  const handleSubmit = (event) => {
    event.preventDefault();
    onSubmit(amount, targetCurrency);
  };

  return (
    <form onSubmit={handleSubmit}>
      <label htmlFor="swap-amount">Swap Amount:</label>
      <input
        type="number"
        id="swap-amount"
        value={amount}
        onChange={(event) => setAmount(event.target.value)}
      />
      <label htmlFor="target-currency">Target Currency:</label>
      <input
        type="text"
        id="target-currency"
        value={targetCurrency}
        onChange={(event) => setTargetCurrency(event.target.value)}
      />
      <button type="submit">Swap Tokens</button>
    </form>
  );
};

export default SwapForm;
