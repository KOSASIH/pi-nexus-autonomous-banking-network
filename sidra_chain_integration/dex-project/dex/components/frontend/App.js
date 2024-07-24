import React, { useState, useEffect } from 'react';
import { connect } from 'react-redux';
import { bindActionCreators } from 'redux';
import { fetchBalance, fetchTransactionHistory } from '../actions';
import BalanceComponent from '../components/BalanceComponent';
import TransactionHistoryComponent from '../components/TransactionHistoryComponent';

const App = ({ balance, transactionHistory, fetchBalance, fetchTransactionHistory }) => {
  const [address, setAddress] = useState('');

  useEffect(() => {
    fetchBalance(address);
    fetchTransactionHistory(address);
  }, [address]);

  const handleAddressChange = (event) => {
    setAddress(event.target.value);
  };

  return (
    <div>
      <h1>SidraChain Wallet</h1>
      <input type="text" value={address} onChange={handleAddressChange} placeholder="Enter your Ethereum address" />
      <BalanceComponent balance={balance} />
      <TransactionHistoryComponent transactionHistory={transactionHistory} />
    </div>
  );
};

const mapStateToProps = (state) => {
  return {
    balance: state.balance,
    transactionHistory: state.transactionHistory,
  };
};

const mapDispatchToProps = (dispatch) => {
  return bindActionCreators({ fetchBalance, fetchTransactionHistory }, dispatch);
};

export default connect(mapStateToProps, mapDispatchToProps)(App);
