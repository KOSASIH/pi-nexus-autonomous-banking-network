import React from 'eact';

function ShipmentList({ shipments }) {
  return (
    <ul>
      {shipments.map((shipment) => (
        <li key={shipment._id}>
          {shipment.sender.name} - {shipment.recipient.name}
        </li>
      ))}
    </ul>
  );
}

export default ShipmentList;
