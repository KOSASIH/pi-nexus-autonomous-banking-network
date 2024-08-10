// Import dependencies
import $ from 'jquery';
import Chart from 'chart.js';
import 'bootstrap/dist/js/bootstrap';

// Dashboard functionality
$(document).ready(() => {
  // Initialize charts
  const ctx = document.getElementById('node-rankings-chart').getContext('2d');
  const nodeRankingsChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: [],
      datasets: [{
        label: 'Node Rankings',
        data: [],
        backgroundColor: [
          'rgba(255, 99, 132, 0.2)',
          'rgba(54, 162, 235, 0.2)',
          'rgba(255, 206, 86, 0.2)',
          'rgba(75, 192, 192, 0.2)',
          'rgba(153, 102, 255, 0.2)',
          'rgba(255, 159, 64, 0.2)'
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(54, 162, 235, 1)',
          'rgba(255, 206, 86, 1)',
          'rgba(75, 192, 192, 1)',
          'rgba(153, 102, 255, 1)',
          'rgba(255, 159, 64, 1)'
        ],
        borderWidth: 1
      }]
    },
    options: {
      scales: {
        y: {
          beginAtZero: true
        }
      }
    }
  });

  const ctx2 = document.getElementById('transaction-activity-chart').getContext('2d');
  const transactionActivityChart = new Chart(ctx2, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Transaction Activity',
        data: [],
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        borderColor: 'rgba(255, 99, 132, 1)',
        borderWidth: 1
      }]
    },
    options: {
      scales: {
        y: {
          beginAtZero: true
        }
      }
    }
  });

  // Load data from API
  $.ajax({
    url: '/api/dashboard',
    method: 'GET',
    dataType: 'json',
    success: (data) => {
      // Update node rankings chart
      nodeRankingsChart.data.labels = data.nodes.map((node) => node.name);
      nodeRankingsChart.data.datasets[0].data = data.nodes.map((node) => node.reputation);
      nodeRankingsChart.update();

      // Update transaction activity chart
      transactionActivityChart.data.labels = data.transactions.map((transaction) => transaction.timestamp);
      transactionActivityChart.data.datasets[0].data = data.transactions.map((transaction) => transaction.amount);
      transactionActivityChart.update();
    }
  });

  // Real-time updates
  setInterval(() => {
    $.ajax({
      url: '/api/dashboard',
      method: 'GET',
      dataType: 'json',
      success: (data) => {
        // Update node rankings chart
        nodeRankingsChart.data.labels = data.nodes.map((node) => node.name);
        nodeRankingsChart.data.datasets[0].data = data.nodes.map((node) => node.reputation);
        nodeRankingsChart.update();

        // Update transaction activity chart
        transactionActivityChart.data.labels = data.transactions.map((transaction) => transaction.timestamp);
        transactionActivityChart.data.datasets[0].data = data.transactions.map((transaction) => transaction.amount);
        transactionActivityChart.update();
      }
    });
  }, 10000);
});
