import { useState } from 'eact';

const VestForm = ({ onSubmit }) => {
  const [amount, setAmount] = useState(0);
  const [vestingPeriod, setVestingPeriod] = useState('');

  const handleSubmit = (event) => {
    event.preventDefault();
    onSubmit(amount, vestingPeriod);
  };

  return (
    <form onSubmit={handleSubmit}>
      <label htmlFor="vest-amount">Vest Amount:</label>
      <input
        type="number"
        id="vest-amount"
        value={amount}
        onChange={(event) => setAmount(event.target.value)}
      />
      <label htmlFor="vesting-period">Vesting Period:</label>
      <input
        type="text"
        id="vesting-period"
        value={vestingPeriod}
        onChange={(event) => setVestingPeriod(event.target.value)}
      />
      <button type="submit">Vest Tokens</button>
    </form>
  );
};

export default VestForm;
