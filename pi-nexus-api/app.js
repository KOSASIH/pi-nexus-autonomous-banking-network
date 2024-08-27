import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import compression from 'compression';
import routes from './routes';
import models from './models';
import utils from './utils';

const app = express();

app.use(helmet());
app.use(cors());
app.use(compression());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use('/v1', routes.v1);
app.use('/v2', routes.v2);

app.use((err, req, res, next) => {
  utils.errorHandler(err, req, res, next);
});

app.listen(process.env.PORT || 3000, () => {
  console.log(`Pi Nexus API listening on port ${process.env.PORT || 3000}`);
});

export default app;
