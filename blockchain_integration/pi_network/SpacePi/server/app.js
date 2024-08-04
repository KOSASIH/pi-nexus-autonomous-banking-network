const express = require('express');
const app = express();
const routes = require('./routes');
const mongoose = require('mongoose');

mongoose.connect('mongodb://localhost/spacepi', { useNewUrlParser: true, useUnifiedTopology: true });

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(routes);

app.listen(3000, () => {
  console.log('Server started on port 3000');
});
