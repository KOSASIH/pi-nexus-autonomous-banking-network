// Import required libraries
import express from 'express';
import axios from 'axios';
import cors from 'cors';
import bodyParser from 'body-parser';

// Define constants
const API_URL = 'https://api.example.com';
const MODEL_URL = 'https://model.example.com';
const PORT = 3000;

// Create an Express app
const app = express();

// Use middleware
app.use(cors());
app.use(bodyParser.json());

// Define routes
app.get('/data', async (req, res) => {
  try {
    const response = await axios.get(API_URL + '/data');
    res.json(response.data);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error fetching data' });
  }
});

app.get('/model', async (req, res) => {
  try {
    const response = await axios.get(MODEL_URL + '/model');
    res.json(response.data);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error fetching model' });
  }
});

app.post('/predict', async (req, res) => {
  try {
    const data = req.body;
    const response = await axios.post(MODEL_URL + '/predict', data);
    res.json(response.data);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error making prediction' });
  }
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server started on port ${PORT}`);
});
