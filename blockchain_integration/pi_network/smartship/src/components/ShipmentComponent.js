import React from 'eact';
import { Grid, Typography, Button } from '@material-ui/core';

const ShipmentComponent = ({ shipment, onUpdate, onDelete }) => {
  const [editing, setEditing] = useState(false);

  const handleEdit = () => {
    setEditing(true);
  };

  const handleSave = () => {
    onUpdate(shipment);
    setEditing(false);
  };

  const handleCancel = () => {
    setEditing(false);
  };

  const handleDelete = () => {
    onDelete(shipment);
  };

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Typography variant="h5">{shipment.sender.name} - {shipment.recipient.name}</Typography>
      </Grid>
      <Grid item xs={6}>
        <Typography variant="body1">Shipment Type: {shipment.shipmentType}</Typography>
      </Grid>
      <Grid item xs={6}>
        <Typography variant="body1">Weight: {shipment.weight} kg</Typography>
      </Grid>
      <Grid item xs={6}>
        <Typography variant="body1">Dimensions: {shipment.dimensions.width} x {shipment.dimensions.height} x {shipment.dimensions.length} cm</Typography>
      </Grid>
      {editing? (
        <Grid item xs={12}>
          <Button variant="contained" color="primary" onClick={handleSave}>
            Save Changes
          </Button>
          <Button variant="contained" color="secondary" onClick={handleCancel}>
            Cancel
          </Button>
        </Grid>
      ) : (
        <Grid item xs={12}>
          <Button variant="contained" color="primary" onClick={handleEdit}>
            Edit
          </Button>
          <Button variant="contained" color="secondary" onClick={handleDelete}>
            Delete
          </Button>
        </Grid>
      )}
    </Grid>
  );
};

export default ShipmentComponent;
