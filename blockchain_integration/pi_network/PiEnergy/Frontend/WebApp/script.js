const registerProducerButton = document.getElementById('register-producer');
const registerConsumerButton = document.getElementById('register-consumer');
const tradeEnergyButton = document.getElementById('trade-energy');

registerProducerButton.addEventListener('click', () => {
    // Call register producer function
    fetch('/api/register-producer', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            address: '0x1234567890abcdef'
        })
    })
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error(error));
});

registerConsumerButton.addEventListener('click', () => {
    // Call register consumer function
    fetch('/api/register-consumer', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            address: '0x9876543210fedcba'
        })
    })
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error(error));
});

tradeEnergyButton.addEventListener('click', () => {
    // Call trade energy function
    fetch('/api/trade-energy', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            producer: '0x1234567890abcdef',
            consumer: '0x9876543210fedcba',
            amount: 100
        })
    })
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error(error));
});
