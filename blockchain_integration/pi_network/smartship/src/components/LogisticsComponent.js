import React, { useState, useEffect } from 'eact';
import { Grid, Typography, Button } from '@material-ui/core';
import ShipmentComponent from './ShipmentComponent';
import axios from 'axios';

const LogisticsComponent = () => {
  const [shipments, setShipments] = useState([]);
  const [newShipment, setNewShipment] = useState({});

  useEffect(() => {
    axios.get('/api/shipments')
     .then(response => {
        setShipments(response.data);
      })
     .catch(error => {
        console.error(error);
      });
  }, []);

  const handleCreateShipment = () => {
    axios.post('/api/shipments', newShipment)
     .then(response => {
        setShipments([...shipments, response.data]);
        setNewShipment({});
      })
     .catch(error => {
        console.error(error);
      });
  };

  const handleUpdateShipment = (shipment) => {
    axios.put(`/api/shipments/${shipment._id}`, shipment)
     .then(response => {
        setShipments(shipments.map(s => s._id === shipment._id? response.data : s));
      })
     .catch(error => {
        console.error(error);
      });
  };

  const handleDeleteShipment = (shipment) => {
    axios.delete(`/api/shipments/${shipment._id}`)
     .then(() => {
        setShipments(shipments.filter(s => s._id!== shipment._id));
      })
     .catch(error => {
        console.error(error);
      });
  };

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Typography variant="h4">Logistics Management</Typography>
      </Grid>
      <Grid item xs={12}>
        <Button variant="contained" color="primary" onClick={handleCreateShipment}>
          Create New Shipment
        </Button>
      </Grid>
      <Grid item xs={12}>
        {shipments.map((shipment) => (
          <ShipmentComponent
            key={shipment._id}
            shipment={shipment}
            onUpdate={handleUpdateShipment}
            onDelete={handleDeleteShipment}
          />
        ))}
      </Grid>
    </Grid>
  );
};

export default LogisticsComponent;
