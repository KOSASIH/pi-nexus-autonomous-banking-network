const axios = require('axios');
const { TELEGRAM_BOT_TOKEN } = require('../config/serverConfig');

const sendTelegramNotification = async (message) => {
    const url = `https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage`;
    try {
        await axios.post(url, {
            chat_id: process.env.TELEGRAM_CHAT_ID, // Set your chat ID
            text: message,
        });
    } catch (error) {
        console.error('Error sending Telegram notification:', error);
    }
};

module.exports = { sendTelegramNotification };
