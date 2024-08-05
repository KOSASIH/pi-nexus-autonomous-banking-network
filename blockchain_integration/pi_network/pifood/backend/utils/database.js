// database.js
const mongoose = require('mongoose');
const config = require('../config');

const connectToDatabase = async () => {
  try {
    await mongoose.connect(config.database.uri, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
      useCreateIndex: true,
      useFindAndModify: false,
    });
    console.log('Connected to database');
  } catch (err) {
    console.error('Error connecting to database:', err);
    process.exit(1);
  }
};

const disconnectFromDatabase = async () => {
  try {
    await mongoose.disconnect();
    console.log('Disconnected from database');
  } catch (err) {
    console.error('Error disconnecting from database:', err);
  }
};

const truncateDatabase = async () => {
  try {
    const collections = await mongoose.connection.db.collections();
    for (const collection of collections) {
      await collection.deleteMany({});
    }
    console.log('Database truncated');
  } catch (err) {
    console.error('Error truncating database:', err);
  }
};

module.exports = {
  connectToDatabase,
  disconnectFromDatabase,
  truncateDatabase,
};
