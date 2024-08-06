import React, { useEffect, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { getHealthRecords } from '../utils/api';
import { setHealthRecords } from '../store/healthRecordsSlice';

const HealthRecordComponent = () => {
  const dispatch = useDispatch();
  const healthRecords = useSelector((state) => state.healthRecords);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await getHealthRecords();
        dispatch(setHealthRecords(response.data));
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
      <h1>Health Records</h1>
      <ul>
        {healthRecords.map((record) => (
          <li key={record.id}>{record.name}</li>
        ))}
      </ul>
    </div>
  );
};

export default HealthRecordComponent;
