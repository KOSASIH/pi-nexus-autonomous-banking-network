const express = require('express');
const app = express();

app.get('/sidra-chain-integration', (req, res) => {
  res.json({ message: 'Sidra Chain Integration Service' });
});

app.listen(8080, () => {
  console.log('Sidra Chain Integration Service listening on port 8080');
});
