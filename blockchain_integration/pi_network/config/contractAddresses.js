// config/contractAddresses.js

const contractAddresses = {
    mainnet: {
        Token: "0xYourMainnetTokenAddress",
        InvestmentPortfolio: "0xYourMainnetInvestmentPortfolioAddress",
    },
    ropsten: {
        Token: "0xYourRopstenTokenAddress",
        InvestmentPortfolio: "0xYourRopstenInvestmentPortfolioAddress",
    },
    rinkeby: {
        Token: "0xYourRinkebyTokenAddress",
        InvestmentPortfolio: "0xYourRinkebyInvestmentPortfolioAddress",
    },
    kovan: {
        Token: "0xYourKovanTokenAddress",
        InvestmentPortfolio: "0xYourKovanInvestmentPortfolioAddress",
    },
    localhost: {
        Token: "0xYourLocalhostTokenAddress",
        InvestmentPortfolio: "0xYourLocalhostInvestmentPortfolioAddress",
    },
};

const getContractAddress = (network, contractName) => {
    return contractAddresses[network]?.[contractName] || null;
};

module.exports = {
    contractAddresses,
    getContractAddress,
};
