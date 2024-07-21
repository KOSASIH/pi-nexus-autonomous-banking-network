import React, { useState } from 'react';

const LoanApplication = () => {
    const [application, setApplication] = useState({
        creditScore: 0,
        income: 0,
        employmentHistory: 0,
        loanAmount: 0
    });

    const handleSubmit = async (event) => {
        event.preventDefault();
        // Call the LoanProcessingContract's evaluateLoanApplication function
        const response = await fetch('/api/loan-processing', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(application)
        });
        const decision = await response.json();
        alert(`Loan application decision: ${decision}`);
    };

    return (
        <form onSubmit={handleSubmit}>
            <label>Credit Score:</label>
            <input type="number" value={application.creditScore} onChange={(event) => setApplication({ ...application, creditScore: event.target.value })} />
            <br />
            <label>Income:</label>
            <input type="number" value={application.income} onChange={(event) => setApplication({ ...application, income: event.target.value })} />
            <br />
            <label>Employment History:</label>
            <input type="number" value={application.employmentHistory} onChange={(event) => setApplication({ ...application, employmentHistory: event.target.value })} />
            <br />
            <label>Loan Amount:</label>
            <input type="number" value={application.loanAmount} onChange={(event) => setApplication({ ...application, loanAmount: event.target.value })} />
            <br />
            <button type="submit">Submit Loan Application</button>
        </form>
    );
};

export default LoanApplication;
