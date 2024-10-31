// config/networkConfig.js

const networks = {
    mainnet: {
        chainId: 1,
        name: "Ethereum Mainnet",
        rpcUrl: "https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID",
        explorerUrl: "https://etherscan.io",
        nativeCurrency: {
            name: "Ether",
            symbol: "ETH",
            decimals: 18,
        },
    },
    ropsten: {
        chainId: 3,
        name: "Ropsten Testnet",
        rpcUrl: "https://ropsten.infura.io/v3/YOUR_INFURA_PROJECT_ID",
        explorerUrl: "https://ropsten.etherscan.io",
        nativeCurrency: {
            name: "Test Ether",
            symbol: "ETH",
            decimals: 18,
        },
    },
    rinkeby: {
        chainId: 4,
        name: "Rinkeby Testnet",
        rpcUrl: "https://rinkeby.infura.io/v3/YOUR_INFURA_PROJECT_ID",
        explorerUrl: "https://rinkeby.etherscan.io",
        nativeCurrency: {
            name: "Test Ether",
            symbol: "ETH",
            decimals: 18,
        },
    },
    kovan: {
        chainId: 42,
        name: "Kovan Testnet",
        rpcUrl: "https://kovan.infura.io/v3/YOUR_INFURA_PROJECT_ID",
        explorerUrl: "https://kovan.etherscan.io",
        nativeCurrency: {
            name: "Test Ether",
            symbol: "ETH",
            decimals: 18,
        },
    },
    localhost: {
        chainId: 1337,
        name: "Localhost",
        rpcUrl: "http://127.0.0.1:8545",
        explorerUrl: "",
        nativeCurrency: {
            name: "Ether",
            symbol: "ETH",
            decimals: 18,
        },
    },
};

const getNetworkConfig = (network) => {
    return networks[network] || networks.localhost;
};

module.exports = {
    networks,
    getNetworkConfig,
};
