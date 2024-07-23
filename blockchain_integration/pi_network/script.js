const apiEndpoint = 'http://localhost:3000/api';

// Get Pi Coin value
fetch(`${apiEndpoint}/pi-coin/value`)
  .then((response) => response.json())
  .then((data) => {
    document.getElementById('pi-coin-value').textContent = `$${data.value}`;
  });

// Cast a vote
document.getElementById('vote-button').addEventListener('click', () => {
  const option = 'Option A'; // Replace with user input
  fetch(`${apiEndpoint}/vote`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ option }),
  })
    .then((response) => response.json())
    .then((data) => {
      console.log(data);
    });
});
