const express = require('express');
const app = express();
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');

mongoose.connect('mongodb://localhost/medical-billing-db', { useNewUrlParser: true, useUnifiedTopology: true });

app.use(express.json());

const User = require('./models/User');
const MedicalBilling = require('./models/MedicalBilling');

app.post('/api/auth/login', async (req, res) => {
  const { username, password } = req.body;
  const user = await User.findOne({ username });
  if (!user) {
    return res.status(401).json({ message: 'Invalid username or password' });
  }
  const isValid = await bcrypt.compare(password, user.password);
  if (!isValid) {
    return res.status(401).json({ message: 'Invalid username or password' });
  }
  const token = jwt.sign({ userId: user._id }, process.env.SECRET_KEY, { expiresIn: '1h' });
  res.json({ token });
});

app.get('/api/medical-billings', async (req, res) => {
  const medicalBillings = await MedicalBilling.find();
  res.json(medicalBillings);
});

app.listen(3001, () => {
  console.log('Server listening on port 3001');
});
