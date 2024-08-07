import express from 'express';
import routes from './routes';
import mongoose from 'mongoose';

const app = express();

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

mongoose.connect('mongodb://localhost/pisure', { useNewUrlParser: true, useUnifiedTopology: true });

app.use('/api', routes);

app.listen(3000, () => {
  console.log('Server listening on port 3000');
});
