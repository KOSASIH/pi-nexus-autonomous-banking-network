// frontend/components/PaymentButton.js
import React, { useState, useEffect } from 'eact';
import { useWeb3 } from '../contexts/Web3Context';
import { usePaymentGateway } from '../contexts/PaymentGatewayContext';
import { ethers } from 'ethers';

const PaymentButton = ({ amount, currency, payer, payee, onPaymentSuccess }) => {
  const { web3, account } = useWeb3();
  const { paymentGatewayContract } = usePaymentGateway();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!web3 ||!paymentGatewayContract) return;
    const getBalance = async () => {
      const balance = await paymentGatewayContract.methods.getBalance(account).call();
      if (balance < amount) {
        setError('Insufficient balance');
      }
    };
    getBalance();
  }, [web3, paymentGatewayContract, account, amount]);

  const handlePayment = async () => {
    if (!web3 ||!paymentGatewayContract) return;
    setLoading(true);
    try {
      const txCount = await web3.eth.getTransactionCount(account);
      const tx = {
        from: account,
        to: paymentGatewayContract.options.address,
        value: ethers.utils.parseEther(amount.toString()),
        gas: '20000',
        gasPrice: ethers.utils.parseUnits('20', 'gwei'),
        nonce: txCount,
      };
      const signedTx = await web3.eth.accounts.signTransaction(tx, account);
      const receipt = await web3.eth.sendTransaction(signedTx.rawTransaction);
      if (receipt.status === '0x1') {
        onPaymentSuccess(receipt.transactionHash);
      } else {
        setError('Payment failed');
      }
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <button
      onClick={handlePayment}
      disabled={loading || error!== null}
      className={`payment-button ${loading? 'loading' : ''}`}
    >
      {loading? 'Processing...' : `Pay ${amount} ${currency}`}
      {error && <div className="error">{error}</div>}
    </button>
  );
};

export default PaymentButton;
