document.getElementById('transaction-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const amount = document.getElementById('amount').value;
    const encrypted_amount = encrypt_data(amount);
    // Send encrypted_amount to the backend for processing
});
