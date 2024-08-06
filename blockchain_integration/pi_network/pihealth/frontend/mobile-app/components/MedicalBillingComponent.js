import React, { useEffect, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { getMedicalBills } from '../utils/api';
import { setMedicalBills } from '../store/medicalBillsSlice';

const MedicalBillingComponent = () => {
  const dispatch = useDispatch();
  const medicalBills = useSelector((state) => state.medicalBills);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await getMedicalBills();
        dispatch(setMedicalBills(response.data));
        setLoading(false);
      } catch (error) {
        setError(error.message);
        setLoading(false);
      }
    };

    fetchData();
  }, [dispatch]);

  if (loading) {
    return <div>Loading...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  return (
    <div>
      <h1>Medical Bills</h1>
      <ul>
        {medicalBills.map((bill) => (
          <li key={bill.id}>{bill.description}</li>
        ))}
      </ul>
    </div>
  );
};

export default MedicalBillingComponent;
