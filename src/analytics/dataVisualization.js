// dataVisualization.js

class DataVisualization {
    constructor(canvasId) {
        this.canvasId = canvasId;
        this.chart = null;
    }

    // Create a bar chart
    createBarChart(labels, data) {
        const ctx = document.getElementById(this.canvasId).getContext('2d');
        this.chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Sentiment Score',
                    data: data,
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1,
                }],
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true,
                    },
                },
            },
        });
    }

    // Update the chart with new data
    updateChart(labels, data) {
        if (this.chart) {
            this.chart.data.labels = labels;
            this.chart.data.datasets[0].data = data;
            this.chart.update();
        } else {
            console.error('Chart not initialized. Please create a chart first.');
        }
    }
}

// Example usage
document.addEventListener('DOMContentLoaded', () => {
    const visualization = new DataVisualization('myChart');

    // Sample data for visualization
    const labels = ['Feedback 1', 'Feedback 2', 'Feedback 3'];
    const data = [0.8, -0.6, 0]; // Example sentiment scores

    visualization.createBarChart(labels, data);
});
