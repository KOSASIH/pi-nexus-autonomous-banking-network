import React from 'react';
import axios from 'axios';

const Dashboard = () => {
  const [medicalBillings, setMedicalBillings] = useState([]);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      axios.get('http://localhost:3001/api/medical-billings', {
        headers: { Authorization: `Bearer ${token}` },
      })
        .then((response) => {
          setMedicalBillings(response.data);
        })
        .catch((error) => {
          console.error(error);
        });
    }
  }, []);

  return (
    <div>
      <h1>Dashboard</h1>
      <ul>
        {medicalBillings.map((medicalBilling) => (
          <li key={medicalBilling._id}>{medicalBilling.patientName}</li>
        ))}
      </ul>
    </div>
  );
};

export default Dashboard;
