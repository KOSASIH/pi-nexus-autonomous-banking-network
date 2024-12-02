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
    } else {
        alert('Please install MetaMask to use this app');
    }
});

// Apply for a loan
document.getElementById('loanForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    const amount = document.getElement ById('amount').value;
    const interestRate = document.getElementById('interestRate').value;
    const duration = document.getElementById('duration').value;

    const accounts = await web3.eth.getAccounts();
    try {
        await loanContract.methods.applyForLoan(amount, interestRate, duration).send({ from: accounts[0] });
        document.getElementById('message').innerText = 'Loan application submitted successfully!';
    } catch (error) {
        document.getElementById('message').innerText = 'Error submitting loan application: ' + error.message;
    }
});

// Check loan status
document.getElementById('checkStatus').addEventListener('click', async () => {
    const loanId = document.getElementById('loanId').value;

    try {
        const loan = await loanContract.methods.getLoanDetails(loanId).call();
        document.getElementById('statusResult').innerText = `Loan Status: ${loan.status}`;
    } catch (error) {
        document.getElementById('statusResult').innerText = 'Error fetching loan status: ' + error.message;
    }
});

// Repay a loan
document.getElementById('repayLoan').addEventListener('click', async () => {
    const loanId = document.getElementById('repayLoanId').value;
    const repayAmount = document.getElementById('repayAmount').value;

    const accounts = await web3.eth.getAccounts();
    try {
        await loanContract.methods.repayLoan(loanId).send({ from: accounts[0], value: repayAmount });
        document.getElementById('message').innerText = 'Loan repaid successfully!';
    } catch (error) {
        document.getElementById('message').innerText = 'Error repaying loan: ' + error.message;
    }
});
