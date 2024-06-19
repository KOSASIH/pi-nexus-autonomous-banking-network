import React, { useState, useEffect } from 'eact';
import { useWeb3React } from '@web3-react/core';
import { useContractLoader, useContractReader } from 'eth-hooks';
import { Grid, Card, CardContent, Typography } from '@material-ui/core';

const SmartShipDashboard = () => {
  const { account, library } = useWeb3React();
  const contract = useContractLoader('SmartShip', library);
  const [shipments, setShipments] = useState([]);
  const [transactions, setTransactions] = useState([]);

  useEffect(() => {
    async function fetchData() {
      const shipments = await contract.methods.getShipments().call();
      const transactions = await contract.methods.getTransactions().call();
      setShipments(shipments);
      setTransactions(transactions);
    }
    fetchData();
  }, [account, library]);

  return (
    <Grid container spacing={2}>
      <Grid item xs={12} sm={6} md={4}>
        <Card>
          <CardContent>
            <Typography variant="h5">Shipments</Typography>
            <ul>
              {shipments.map((shipment, index) => (
                <li key={index}>{shipment.id} - {shipment.status}</li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} sm={6} md={4}>
        <Card>
          <CardContent>
            <Typography variant="h5">Transactions</Typography>
            <ul>
              {transactions.map((transaction, index) => (
                <li key={index}>{transaction.id} - {transaction.amount} ETH</li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default SmartShipDashboard;
