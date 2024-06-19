import React, { useState, useEffect } from 'eact';
import axios from 'axios';
import ShipmentList from './ShipmentList';
import ShipmentForm from './ShipmentForm';

function App() {
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

  const handleSubmit = (event) => {
    event.preventDefault();
    axios.post('/api/shipments', newShipment)
     .then(response => {
        setShipments([...shipments, response.data]);
        setNewShipment({});
      })
     .catch(error => {
        console.error(error);
      });
  };

  const handleInputChange = (event) => {
    setNewShipment({...newShipment, [event.target.name]: event.target.value });
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Logistics Management App</h1>
      </header>
      <main className="app-content">
        <ShipmentList shipments={shipments} />
        <ShipmentForm
          newShipment={newShipment}
          handleInputChange={handleInputChange}
          handleSubmit={handleSubmit}
        />
      </main>
    </div>
  );
}

export default App;
