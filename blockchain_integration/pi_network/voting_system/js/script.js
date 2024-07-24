const votes = {
  total: 0,
  count: 0,
};

function countVote(vote) {
  votes.total += vote;
  votes.count++;
  const globalValue = votes.total / votes.count; // calculate new global value
  document.getElementById('global-value').innerHTML =
    `$${globalValue.toFixed(2)}`;
  document.getElementById('vote-count').innerHTML = `${votes.count} votes cast`;
}

document.getElementById('vote-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const vote = parseInt(document.getElementById('vote').value);
  if (isNaN(vote) || vote < 0) {
    alert('Invalid vote. Please enter a positive number.');
    return;
  }
  countVote(vote);
  document.getElementById('vote').value = ''; // reset input field
});
