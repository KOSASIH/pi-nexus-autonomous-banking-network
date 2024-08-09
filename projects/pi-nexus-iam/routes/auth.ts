import express, { Request, Response, NextFunction } from 'express';
import bcrypt from 'bcrypt';
import jwt from 'jsonwebtoken';
import { User } from '../models/user';
import { Role } from '../enums/role.enum';

const router = express.Router();

router.post('/register', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { username, email, password } = req.body;
    const user = new User({ username, email, password, roles: [Role.USER] });
    await user.save();
    res.json({ message: 'User created successfully' });
  } catch (error) {
    next(error);
  }
});

router.post('/login', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { email, password } = req.body;
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(401).json({ error: 'Invalid email or password' });
    }
    const isValid = await user.comparePassword(password);
    if (!isValid) {
      return res.status(401).json({ error: 'Invalid email or password' });
    }
    const token = user.generateToken();
    res.json({ token });
  } catch (error) {
    next(error);
  }
});

router.post('/forgot-password', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { email } = req.body;
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    const token = await user.generatePasswordResetToken();
    res.json({ token });
  } catch (error) {
    next(error);
  }
});

router.post('/reset-password', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { token, password } = req.body;
    const user = await User.findOne({ passwordResetToken: token });
    if (!user) {
      return res.status(404).json({ error: 'Invalid token' });
    }
    user.password = password;
    await user.save();
    res.json({ message: 'Password reset successfully' });
  } catch (error) {
    next(error);
  }
});

export default router;
