import express from 'express';
import fiatOnRampRouter from './fiatOnRamp';
import walletRouter from './wallet';
import dashboardRouter from './dashboard';

const app = express();

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use('/fiat-on-ramp', fiatOnRampRouter);
app.use('/wallet', walletRouter);
app.use('/dashboard', dashboardRouter);

app.get('/', (req, res) => {
  res.send('OnRampPi API');
});

app.use((err, req, res, next) => {
  console.error(err);
  res.status(500).send('Internal Server Error');
});

export default app;
