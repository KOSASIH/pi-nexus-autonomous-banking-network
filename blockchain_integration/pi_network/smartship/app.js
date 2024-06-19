const express = require('express');
const app = express();
const cors = require('cors');
const helmet = require('helmet');
const mongoose = require('mongoose');
const { shipmentRouter } = require('./routes/shipment');
const { authRouter } = require('./routes/auth');
const { errorHandlingMiddleware } = require('./middleware/errorHandling');

mongoose.connect('mongodb://localhost/logistics', { useNewUrlParser: true, useUnifiedTopology: true });

app.use(cors());
app.use(helmet());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use('/api/shipments', shipmentRouter);
app.use('/api/auth', authRouter);

app.use(errorHandlingMiddleware);

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server started on port ${port}`);
});
