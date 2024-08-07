import React from 'react';

const PolicyCard = ({ policy }) => {
  return (
    <div className="policy-card">
      <h2>{policy.policyType}</h2>
      <p>Policy Holder: {policy.firstName} {policy.lastName}</p>
      <p>Email: {policy.email}</p>
      <p>Policy ID: {policy.policyId}</p>
    </div>
  );
};

export default PolicyCard;
