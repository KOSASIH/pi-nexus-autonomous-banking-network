// script.js
const createAccountForm = document.getElementById('create-account-form');
const depositFundsForm = document.getElementById('deposit-funds-form');
const withdrawFundsForm = document.getElementById('withdraw-funds-form');
const getBalanceForm = document.getElementById('get-balance-form');
const resultDiv = document.getElementById('result');

createAccountForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const userIdentity = document.getElementById('user-identity').value;
  const response = await fetch('/create-account', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ userIdentity }),
  });
  const data = await response.json();
  resultDiv.innerHTML = `Account created successfully! Account address: ${data.accountAddress}`;
});

depositFundsForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const accountAddress = document.getElementById('account-address').value;
  const amount = document.getElementById('amount').value;
  const response = await fetch('/deposit-funds', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ accountAddress, amount }),
  });
  const data = await response.json();
  resultDiv.innerHTML = `Funds deposited successfully! New balance: ${data.balance}`;
});

withdrawFundsForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const accountAddress = document.getElementById('account-address').value;
  const amount = document.getElementById('amount').value;
  const response = await fetch('/withdraw-funds', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ accountAddress, amount }),
  });
  const data = await response.json();
  resultDiv.innerHTML = `Funds withdrawn successfully! New balance: ${data.balance}`;
});

getBalanceForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const accountAddress = document.getElementById('account-address').value;
  const response = await fetch('/get-balance', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ accountAddress }),
  });
  const data = await response.json();
  resultDiv.innerHTML = `Account balance: ${data.balance}`;
});
