const express = require('express');
const app = express();

app.use(express.json());

app.post('/api/login', (req, res) => {
  const { username, password } = req.body;
  // Implement login logic
  res.json({ message: 'Login successful' });
});

app.listen(3000, () => {
  console.log('API Gateway listening on port 3000');
});
