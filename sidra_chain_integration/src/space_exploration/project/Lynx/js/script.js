const form = document.getElementById('predict-form');
const resultsDiv = document.getElementById('results');

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const data = document.getElementById('data').value;
    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: data })
    });
    const threats = await response.json();
    const threatsList = threats.threats;
    const threatsHtml = threatsList.map((threat) => {
        return `<p>Threat detected: ${threat}</p>`;
    }).join('');
    resultsDiv.innerHTML = threatsHtml;
});

// Add event listener to the clear button
const clearButton = document.getElementById('clear-button');
clearButton.addEventListener('click', () => {
    document.getElementById('data').value = '';
    resultsDiv.innerHTML = '';
});

// Add event listener to the example button
const exampleButton = document.getElementById('example-button');
exampleButton.addEventListener('click', () => {
    const exampleData = `{
        "src_ip": "192.168.1.1",
        "dst_ip": "8.8.8.8",
        "protocol": "TCP",
        "packet_size": 100
    }`;
    document.getElementById('data').value = exampleData;
});
