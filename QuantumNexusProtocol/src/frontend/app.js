document.addEventListener('DOMContentLoaded', () => {
    console.log("Blockchain Dashboard Loaded");

    // Initialize Web3
    if (typeof window.ethereum !== 'undefined') {
        const web3 = new Web3(window.ethereum);
        console.log("Web3 is enabled");
    } else {
        console.log("Please install MetaMask!");
    }

    // Example function to connect wallet
    document.getElementById('connectWallet').addEventListener('click', async () => {
        try {
            const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
            console.log("Connected account:", accounts[0]);
        } catch (error) {
            console.error("Error connecting wallet:", error);
        }
    });
});
