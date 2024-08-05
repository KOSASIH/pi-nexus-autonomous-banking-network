const express = require('express');
const app = express();
const routes = require('./routes');

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use('/api', routes);

app.listen(config.server.port, () => {
  console.log(`Server listening on port ${config.server.port}`);
});
