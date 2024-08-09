import express, { Request, Response } from 'express';
import { authController } from './controllers/auth';
import { userController } from './controllers/users';

const app = express();

app.use(express.json());
app.use('/api/auth', authController);
app.use('/api/users', userController);

app.listen(3000, () => {
  console.log('Server listening on port 3000');
});
