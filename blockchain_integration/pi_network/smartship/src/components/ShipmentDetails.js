import React, { useState, useEffect } from 'eact';
import { useWeb3React } from '@web3-react/core';
import { useContractLoader, useContractReader } from 'eth-hooks';
import { Grid, Card, CardContent, Typography } from '@material-ui/core';

const ShipmentDetails = ({ shipmentId }) => {
  const { account, library } = useWeb3React();
  const contract = useContractLoader('SmartShip', library);
  const [shipment, setShipment] = useState({});

  useEffect(() => {
    async function fetchData() {
      const shipment = await contract.methods.getShipment(shipmentId).call();
      setShipment(shipment);
    }
    fetchData();
  }, [shipmentId, account, library]);

  return (
    <Grid container spacing={2}>
      <Grid item xs={12} sm={6} md={4}>
        <Card>
          <CardContent>
            <Typography variant="h5">Shipment Details</Typography>
            <ul>
              <li>ID: {shipment.id}</li>
              <li>Status: {shipment.status}</li>
              <li>Sender: {shipment.sender}</li>
              <li>Receiver: {shipment.receiver}</li>
              <li>Amount: {shipment.amount} ETH</li>
            </ul>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default ShipmentDetails;
