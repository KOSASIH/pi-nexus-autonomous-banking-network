import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import compression from 'compression';
import morgan from 'morgan';
import { Scalability } from './scalability/scalability';
import { Utils } from './utils/utils';

const app = express();

app.use(helmet());
app.use(cors());
app.use(compression());
app.use(morgan('dev'));

const scalability = new Scalability();
const utils = new Utils();

app.get('/', (req, res) => {
  res.send('Welcome to the most advanced high-tech API!');
});

app.post('/scale-up', async (req, res) => {
  try {
    await scalability.scaleUp();
    res.send('Scaled up successfully!');
  } catch (error) {
    console.error(error);
    res.status(500).send('Error scaling up');
  }
});

app.post('/scale-down', async (req, res) => {
  try {
    await scalability.scaleDown();
    res.send('Scaled down successfully!');
  } catch (error) {
    console.error(error);
    res.status(500).send('Error scaling down');
  }
});

app.get('/performance-metrics', async (req, res) => {
  try {
    const metrics = await scalability.getPerformanceMetrics();
    res.json(metrics);
  } catch (error) {
    console.error(error);
    res.status(500).send('Error getting performance metrics');
  }
});

app.post('/cache-data', async (req, res) => {
  try {
    const dataToCache = req.body;
    await scalability.cacheData(dataToCache);
    res.send('Data cached successfully!');
  } catch (error) {
    console.error(error);
    res.status(500).send('Error caching data');
  }
});

app.post('/invalidate-cache', async (req, res) => {
  try {
    const dataToInvalidate = req.body;
    await scalability.invalidateCache(dataToInvalidate);
    res.send('Cache invalidated successfully!');
  } catch (error) {
    console.error(error);
    res.status(500).send('Error invalidating cache');
  }
});

app.listen(3000, () => {
  console.log('Server listening on port 3000');
});
