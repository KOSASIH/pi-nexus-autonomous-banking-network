import React, { useState } from 'eact';
import { connect } from 'eact-redux';
import { bindActionCreators } from 'edux';
import * as shipmentActions from '../actions/shipmentActions';
import ShipmentComponent from '../components/ShipmentComponent';

const ShipmentContainer = ({ shipment, actions }) => {
  const [editing, setEditing] = useState(false);

  const handleEdit = () => {
    setEditing(true);
  };

  const handleSave = () => {
    actions.updateShipment(shipment);
    setEditing(false);
  };

  const handleCancel = () => {
    setEditing(false);
  };

  const handleDelete = () => {
    actions.deleteShipment(shipment);
  };

  return (
    <ShipmentComponent
      shipment={shipment}
      onEdit={handleEdit}
      onSave={handleSave}
      onCancel={handleCancel}
      onDelete={handleDelete}
    />
  );
};

const mapStateToProps = (state, ownProps) => {
  return {
    shipment: state.shipments.find((s) => s._id === ownProps.match.params.id),
  };
};

const mapDispatchToProps = (dispatch) => {
  return {
    actions: bindActionCreators(shipmentActions, dispatch),
  };
};

export default connect(mapStateToProps, mapDispatchToProps)(ShipmentContainer);
