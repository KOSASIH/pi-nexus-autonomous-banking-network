// index.js
const userForm = document.getElementById('user-form');
const verifyButton = document.getElementById('verify-button');
const resultElement = document.getElementById('result');

verifyButton.addEventListener('click', async (e) => {
  e.preventDefault();
  const userAddress = document.getElementById('user-address').value;
  const userData = document.getElementById('user-image').files[0];

  // Call the backend API to verify the user's identity
  const response = await fetch('/api/verify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ userAddress, userData }),
  });

  const result = await response.json();
  resultElement.innerText = `User ${userAddress} is ${result.verified ? 'verified' : 'not verified'}`;
});
