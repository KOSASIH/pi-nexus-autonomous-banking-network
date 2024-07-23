// Get the current Pi coin value
function getPiCoinValue() {
  return 314.159; // Replace with API call or other method to get the current value
}

// Display the Pi coin value on the dashboard
const piCoinValueElement = document.getElementById('pi-coin-value-text');
piCoinValueElement.textContent = `$${getPiCoinValue()}`;

// Add event listener to the vote button
const voteButton = document.getElementById('vote-button');
voteButton.addEventListener('click', () => {
  // Cast a vote with the current Pi coin value
  castVote(getPiCoinValue());
});

// Function to cast a vote
function castVote(value) {
  // Implement the voting logic here, e.g., send a request to the backend API
  console.log(`Vote cast with value: ${value}`);
}
