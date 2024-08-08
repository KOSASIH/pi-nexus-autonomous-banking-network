const express = require('express');
const app = express();
const bodyParser = require('body-parser');
const { OracleNexus } = require('../contracts/OracleNexus');

app.use(bodyParser.json());

const oracleNexus = new OracleNexus();

// Ingest data endpoint
app.post('/ingest', async (req, res) => {
  const { data } = req.body;
  try {
    // Validate data
    const isValid = await oracleNexus.validateData(data);
    if (!isValid) {
      return res.status(400).json({ error: 'Invalid data' });
    }

    // Encrypt data
    const encryptedData = await oracleNexus.encryptData(data);

    // Send data to Oracle Nexus
    const requestId = await oracleNexus.sendRequest(encryptedData);

    res.json({ requestId });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Error ingesting data' });
  }
});

// Get data endpoint
app.get('/data', async (req, res) => {
  try {
    const data = await oracleNexus.getData();
    res.json(data);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Error getting data' });
  }
});

app.listen(3000, () => {
  console.log('Data ingestion API listening on port 3000');
});
