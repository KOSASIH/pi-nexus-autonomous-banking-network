import express from 'express';
import apiRouter from './routes/api';

const app = express();

app.use(express.json());
app.use('/api', apiRouter);

app.listen(3000, () => {
  console.log('Server listening on port 3000');
});
