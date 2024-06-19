import React, { useState, useEffect } from 'eact';
import { connect } from 'eact-redux';
import { bindActionCreators } from 'edux';
import * as logisticsActions from '../actions/logisticsActions';
import LogisticsComponent from '../components/LogisticsComponent';

const LogisticsContainer = ({ logistics, actions }) => {
  const [newShipment, setNewShipment] = useState({});

  useEffect(() => {
    actions.fetchShipments();
  }, []);

  const handleCreateShipment = () => {
    actions.createShipment(newShipment);
    setNewShipment({});
  };

  const handleUpdateShipment = (shipment) => {
    actions.updateShipment(shipment);
  };

  const handleDeleteShipment = (shipment) => {
    actions.deleteShipment(shipment);
  };

  return (
    <LogisticsComponent
      shipments={logistics.shipments}
      onCreate={handleCreateShipment}
      onUpdate={handleUpdateShipment}
      onDelete={handleDeleteShipment}
    />
  );
};

const mapStateToProps = (state) => {
  return {
    logistics: state.logistics,
  };
};

const mapDispatchToProps = (dispatch) => {
  return {
    actions: bindActionCreators(logisticsActions, dispatch),
  };
};

export default connect(mapStateToProps, mapDispatchToProps)(LogisticsContainer);
