import React, { useState, useEffect } from 'eact';
import { useWeb3React } from '@web3-react/core';
import { useContractLoader, useContractReader } from 'eth-hooks';
import { Grid, Card, CardContent, Typography } from '@material-ui/core';

const TransactionDetails = ({ transactionId }) => {
  const { account, library } = useWeb3React();
  const contract = useContractLoader('SmartShip', library);
  const [transaction, setTransaction] = useState({});

  useEffect(() => {
    async function fetchData() {
      const transaction = await contract.methods.getTransaction(transactionId).call();
      setTransaction(transaction);
    }
    fetchData();
  }, [transactionId, account, library]);

  return (
   <Grid container spacing={2}>
      <Grid item xs={12} sm={6} md={4}>
        <Card>
          <CardContent>
            <Typography variant="h5">Transaction Details</Typography>
            <ul>
              <li>ID: {transaction.id}</li>
              <li>Sender: {transaction.sender}</li>
              <li>Receiver: {transaction.receiver}</li>
              <li>Amount: {transaction.amount} ETH</li>
              <li>Timestamp: {transaction.timestamp}</li>
            </ul>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default TransactionDetails;
