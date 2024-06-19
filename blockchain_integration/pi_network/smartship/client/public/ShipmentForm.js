import React from 'eact';

function ShipmentForm({ newShipment, handleInputChange, handleSubmit }) {
  return (
    <form onSubmit={handleSubmit}>
      <label>
        Sender Name:
        <input type="text" name="senderName" value={newShipment.senderName} onChange={handleInputChange} />
      </label>
      <label>
        Recipient Name:
        <input type="text" name="recipientName" value={newShipment.recipientName} onChange={handleInputChange} />
      </label>
      <label>
        Shipment Type:
        <select name="shipmentType" value={newShipment.shipmentType} onChange={handleInputChange}>
          <option value="package">Package</option>
          <option value="letter">Letter</option>
        </select>
      </label>
      <button type="submit">Create Shipment</button>
    </form>
  );
}

export default ShipmentForm;
