const axios = require('axios');

const getPrice = async (currency) => {
    try {
        const response = await axios.get(`https://api.coingecko.com/api/v3/simple/price?ids=${currency}&vs_currencies=usd`);
        return response.data[currency].usd;
    } catch (error) {
        throw new Error('Error fetching price data');
    }
};

module.exports = { getPrice };
