require('dotenv').config();

const PORT = process.env.PORT || 3000;
const MONGO_URI = process.env.MONGO_URI;
const JWT_SECRET = process.env.JWT_SECRET;

module.exports = {
    PORT,
    MONGO_URI,
    JWT_SECRET,
};
