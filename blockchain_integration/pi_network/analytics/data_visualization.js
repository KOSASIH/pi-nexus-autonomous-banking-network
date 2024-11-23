// Fetch data from the Flask API and visualize it using Chart.js
async function fetchData() {
    const response = await fetch('/api/data');
    const data = await response.json();
    return data;
}

async function renderChart() {
    const data = await fetchData();
    
    const labels = data.map(entry => entry.timestamp);
    const values = data.map(entry => entry.value);

    const ctx = document.getElementById('myChart').getContext('2d');
    const myChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Sample Data',
                data: values,
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 2,
                fill: false
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day'
                    }
                },
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

document.addEventListener('DOMContentLoaded', renderChart);
