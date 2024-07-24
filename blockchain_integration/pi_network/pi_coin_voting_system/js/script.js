const votes = [];
const piValue = 314.159;

function castVote(vote) {
  votes.push(vote);
  const voteCount = votes.length;
  const voteAverage = votes.reduce((a, b) => a + b, 0) / voteCount;
  const voteStandardDeviation = Math.sqrt(votes.reduce((a, b) => a + Math.pow(b - voteAverage, 2), 0) / voteCount);
  document.getElementById("vote-count").innerHTML = `${voteCount} votes cast`;
  document.getElementById("pi-value").innerHTML = `$${piValue}`;
  document.getElementById("vote-average").innerHTML = `Average vote: $${voteAverage.toFixed(2)}`;
  document.getElementById("vote-standard-deviation").innerHTML = `Standard deviation: $${voteStandardDeviation.toFixed(2)}`;
}

document.getElementById("vote-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const vote = parseFloat(document.getElementById("vote").value);
  if (isNaN(vote) || vote < 0) {
    alert("Invalid vote. Please enter a positive number.");
    return;
  }
  castVote(vote);
  document.getElementById("vote").value = "";
});

document.getElementById("vote").addEventListener("input", () => {
  const vote = parseFloat(document.getElementById("vote").value);
  if (vote < 0) {
    document.getElementById("vote").setCustomValidity("Please enter a positive number.");
  } else {
    document.getElementById("vote").setCustomValidity("");
  }
});
