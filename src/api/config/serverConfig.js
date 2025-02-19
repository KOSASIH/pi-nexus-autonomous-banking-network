require('dotenv').config();

const PORT = process.env.PORT || 3000;
const MONGO_URI = process.env.MONGO_URI;
const TELEGRAM_BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;

module.exports = {
    PORT,
    MONGO_URI,
    TELEGRAM_BOT_TOKEN,
};
