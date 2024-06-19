import React from 'eact';
import { Link } from 'eact-router-dom';
import { Card, CardContent, CardMedia, Typography, Button } from '@material-ui/core';
import { makeStyles } from '@material-ui/core/styles';

const useStyles = makeStyles((theme) => ({
  card: {
    maxWidth: 345,
    margin: theme.spacing(2),
  },
  media: {
    height: 140,
  },
  button: {
    margin: theme.spacing(1),
  },
}));

function ShipmentCard({ shipment }) {
  const classes = useStyles();

  return (
    <Card className={classes.card}>
      <CardMedia
        className={classes.media}
        image={shipment.image}
        title={shipment.sender.name}
      />
      <CardContent>
        <Typography gutterBottom variant="h5" component="h2">
          {shipment.sender.name} - {shipment.recipient.name}
        </Typography>
        <Typography variant="body2" color="textSecondary" component="p">
          Shipment Type: {shipment.shipmentType}
        </Typography>
        <Typography variant="body2" color="textSecondary" component="p">
          Weight: {shipment.weight} kg
        </Typography>
        <Typography variant="body2" color="textSecondary" component="p">
          Dimensions: {shipment.dimensions.width} x {shipment.dimensions.height} x {shipment.dimensions.length} cm
        </Typography>
      </CardContent>
      <Button
        variant="contained"
        color="primary"
        className={classes.button}
        component={Link}
        to={`/shipments/${shipment._id}`}
      >
        View Details
      </Button>
    </Card>
  );
}

export default ShipmentCard;
