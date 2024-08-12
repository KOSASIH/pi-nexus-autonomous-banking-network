import express, { Application } from 'express';
import helmet from 'helmet';
import cors from 'cors';
import mongoose from 'mongoose';
import { API_URL, API_VERSION } from './utils/constants';
import rideRoutes from './routes/ride';
import userRoutes from './routes/user';
import authRoutes from './routes/auth';

const app: Application = express();

app.use(helmet());
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

mongoose.connect(MONGODB_URI, { useNewUrlParser: true, useUnifiedTopology: true });

app.use(`${API_URL}/${API_VERSION}/rides`, rideRoutes);
app.use(`${API_URL}/${API_VERSION}/users`, userRoutes);
app.use(`${API_URL}/${API_VERSION}/auth`, authRoutes);

app.listen(3000, () => {
  console.log(`PiRide API listening on port 3000`);
});
