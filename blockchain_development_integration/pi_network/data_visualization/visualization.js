// Ensure you have the correct contract address and ABI
const contractAddress = "YOUR_CONTRACT_ADDRESS_HERE";
const contractABI = [
    // Add your contract ABI here
];

let web3;
let loanContract;

window.addEventListener('load', async () => {
    // Check if Web3 is injected (MetaMask)
    if (typeof window.ethereum !== 'undefined') {
        web3 = new Web3(window.ethereum);
        await window.ethereum.request({ method: 'eth_requestAccounts' });
        loanContract = new web3.eth.Contract(contractABI, contractAddress);
        await loadLoanData();
    } else {
        alert('Please install MetaMask to use this app');
    }
});

// Load loan data and render the chart
async function loadLoanData() {
    const loanCount = await loanContract.methods.loanCounter().call();
    const loanAmounts = [];
    const loanStatuses = [];

    for (let i = 1; i <= loanCount; i++) {
        const loan = await loanContract.methods.getLoanDetails(i).call();
        loanAmounts.push(loan.amount);
        loanStatuses.push(loan.status);
    }

    renderChart(loanAmounts, loanStatuses);
}

// Render the chart using Chart.js
function renderChart(loanAmounts, loanStatuses) {
    const ctx = document.getElementById('loanChart').getContext('2d');
    const loanChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: loanStatuses,
            datasets: [{
                label: 'Loan Amounts',
                data: loanAmounts,
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
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
}
