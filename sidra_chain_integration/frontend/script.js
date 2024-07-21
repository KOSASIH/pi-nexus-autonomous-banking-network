const loanApplicationForm = document.querySelector('#loan-application form');

loanApplicationForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const creditScore = document.querySelector('#credit-score').value;
    const income = document.querySelector('#income').value;
    const employmentHistory = document.querySelector('#employment-history').value;
    const loanAmount = document.querySelector('#loan-amount').value;

    try {
        const response = await fetch('/api/loan-processing', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ creditScore, income, employmentHistory, loanAmount }),
        });

        const data = await response.json();
        console.log(data);
    } catch (error) {
        console.error(error);
    }
});
