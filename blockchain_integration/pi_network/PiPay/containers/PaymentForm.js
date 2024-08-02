// containers/PaymentForm.js
import React, { useState, useEffect } from 'react';
import { useWeb3 } from '../contexts/Web3Context';
import { usePaymentGateway } from '../contexts/PaymentGatewayContext';
import PaymentButton from '../components/PaymentButton';
import { ethers } from 'ethers';

const PaymentForm = () => {
  const { web3, account } = useWeb3();
  const { paymentGatewayContract } = usePaymentGateway();
  const [amount, setAmount] = useState('');
  const [currency, setCurrency] = useState('ETH');
  const [payer, setPayer] = useState(account);
  const [payee, setPayee] = useState('');
  const [paymentMethod, setPaymentMethod] = useState('contract');
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!web3 ||!paymentGatewayContract) return;
    const getMerchantAddress = async () => {
      const merchantAddress = await paymentGatewayContract.methods.getMerchantAddress().call();
      setPayee(merchantAddress);
    };
    getMerchantAddress();
  }, [web3, paymentGatewayContract]);

  const handleAmountChange = (event) => {
    setAmount(event.target.value);
  };

  const handleCurrencyChange = (event) => {
    setCurrency(event.target.value);
  };

  const handlePaymentMethodChange = (event) => {
    setPaymentMethod(event.target.value);
  };

  const handlePaymentSuccess = (transactionHash) => {
    console.log(`Payment successful! Transaction hash: ${transactionHash}`);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    if (paymentMethod === 'contract') {
      // Process payment using the Payment Gateway contract
      paymentGatewayContract.methods.processPayment(payer, payee, ethers.utils.parseEther(amount.toString()))
        .send({ from: payer })
        .on('transactionHash', handlePaymentSuccess)
        .on('error', (error) => setError(error.message));
    } else {
      // Process payment using a different method (e.g. Metamask)
      // ...
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <label>
        Amount:
        <input type="number" value={amount} onChange={handleAmountChange} />
      </label>
      <label>
        Currency:
        <select value={currency} onChange={handleCurrencyChange}>
          <option value="ETH">ETH</option>
          <option value="USD">USD</option>
          <option value="EUR">EUR</option>
        </select>
      </label>
      <label>
        Payment Method:
        <select value={paymentMethod} onChange={handlePaymentMethodChange}>
          <option value="contract">Payment Gateway Contract</option>
          <option value="metamask">Metamask</option>
        </select>
      </label>
      <PaymentButton
        amount={amount}
        currency={currency}
        payer={payer}
        payee={payee}
        onPaymentSuccess={handlePaymentSuccess}
      />
      {error && <div className="error">{error}</div>}
    </form>
  );
};

export default PaymentForm;
