// dashboard.js

document.addEventListener('DOMContentLoaded', function() {
    // Example: Fetch data from an API and display it on the dashboard
    fetchData();

    // Event listener for a button click
    const refreshButton = document.getElementById('refreshButton');
    refreshButton.addEventListener('click', fetchData);
});

function fetchData() {
    // Simulate fetching data from an API
    console.log('Fetching data...');

    // Example data
    const data = [
        { id: 1, name: 'Loan Application 1', status: 'Pending' },
        { id: 2, name: 'Loan Application 2', status: 'Approved' },
        { id: 3, name: 'Loan Application 3', status: 'Rejected' }
    ];

    displayData(data);
}

function displayData(data) {
    const dataContainer = document.getElementById('dataContainer');
    dataContainer.innerHTML = ''; // Clear previous data

    data.forEach(item => {
        const card = document.createElement('div');
        card.className = 'card';
        card.innerHTML = `
            <h2>${item.name}</h2>
            <p>Status: ${item.status}</p>
        `;
        dataContainer.appendChild(card);
    });
}
