const express = require('express');
const bodyParser = require('body-parser');
const csvParser = require('csv-parser');
const { KDTree } = require('kd-tree-js');

const app = express();
app.use(bodyParser.json());

const data = [];
const kdTree = new KDTree(data, 2);

const loadData = async () => {
  try {
    const response = await axios.get('data.csv', { responseType: 'stream' });
    response.data.pipe(csvParser()).on('data', (row) => {
      data.push(row);
      kdTree.insert(row);
    });
  } catch (error) {
    console.error('Failed to load data:', error.message);
  }
};

const predict = (req, res) => {
  const { x, y } = req.body;
  const result = kdTree.search(x, y);
  res.json(result);
};

app.post('/predict', predict);

loadData();
app.listen(3000, () => {
  console.log('Analytics dashboard listening on port 3000...');
});
