const mongoose = require('mongoose');
const { connectDB } = require('../src/api/config/dbConfig');

beforeAll(async () => {
    await connectDB();
});

afterAll(async () => {
    await mongoose.connection.close();
});
