import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import logger from './utils/logger';
import apiRouter from './routes/api';

const app = express();

app.use(cors());
app.use(helmet());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use('/api', apiRouter);

app.listen(3001, () => {
  logger.info('Server started on port 3001');
});
