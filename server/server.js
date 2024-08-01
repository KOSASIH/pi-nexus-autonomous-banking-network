// server.js
const express = require('express');
const app = express();
const bodyParser = require('body-parser');

app.use(bodyParser.json());

// API endpoints will go here

app.listen(3000, () => {
  console.log('Server listening on port 3000');
});
