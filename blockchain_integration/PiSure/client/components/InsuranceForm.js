import React, { useState } from 'react';
import { useForm } from 'react-hook-form';

const InsuranceForm = () => {
  const { register, handleSubmit, errors } = useForm();
  const [policyData, setPolicyData] = useState({});

  const onSubmit = async (data) => {
    // Call API to create new policy
    const response = await fetch('/api/policies', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });

    if (response.ok) {
      setPolicyData(data);
    } else {
      console.error('Error creating policy:', response.status);
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <label>
        First Name:
        <input type="text" {...register('firstName')} />
      </label>
      <label>
        Last Name:
        <input type="text" {...register('lastName')} />
      </label>
      <label>
        Email:
        <input type="email" {...register('email')} />
      </label>
      <label>
        Policy Type:
        <select {...register('policyType')}>
          <option value="health">Health</option>
          <option value="life">Life</option>
          <option value="auto">Auto</option>
        </select>
      </label>
      <button type="submit">Create Policy</button>
    </form>
  );
};

export default InsuranceForm;
