import { Request, Response } from 'express';
import { authenticate } from '../services/auth.service';
import { validate } from '../utils/validation.util';
import { AuthRequest } from '../types/auth.type';

const authController = {
  async login(req: Request, res: Response) {
    try {
      const { username, password } = req.body;
      const user = await authenticate(username, password);
      const token = await generateToken(user);
      res.json({ token });
    } catch (error) {
      res.status(401).json({ error: 'Invalid credentials' });
    }
  },

  async register(req: Request, res: Response) {
    try {
      const { username, email, password } = req.body;
      const user = await createUser(username, email, password);
      res.json({ message: 'User created successfully' });
    } catch (error) {
      res.status(400).json({ error: 'Failed to create user' });
    }
  },

  async forgotPassword(req: Request, res: Response) {
    try {
      const { email } = req.body;
      const user = await getUserByEmail(email);
      if (!user) {
        res.status(404).json({ error: 'User not found' });
      } else {
        const token = await generateResetToken(user);
        res.json({ token });
      }
    } catch (error) {
      res.status(500).json({ error: 'Failed to send reset password email' });
    }
  },

  async resetPassword(req: Request, res: Response) {
    try {
      const { token, password } = req.body;
      const user = await getUserByResetToken(token);
      if (!user) {
        res.status(404).json({ error: 'Invalid reset token' });
      } else {
        await updateUserPassword(user, password);
        res.json({ message: 'Password reset successfully' });
      }
    } catch (error) {
      res.status(500).json({ error: 'Failed to reset password' });
    }
  },
};

export default authController;

async function generateToken(user: any) {
  const token = jwt.sign({ userId: user.id }, process.env.SECRET_KEY, {
    expiresIn: '1h',
  });
  return token;
}

async function createUser(username: string, email: string, password: string) {
  const user = new User({ username, email, password });
  await user.save();
  return user;
}

async function getUserByEmail(email: string) {
  const user = await User.findOne({ email });
  return user;
}

async function generateResetToken(user: any) {
  const token = jwt.sign({ userId: user.id }, process.env.SECRET_KEY, {
    expiresIn: '1h',
  });
  await user.updateOne({ resetToken: token });
  return token;
}

async function getUserByResetToken(token: string) {
  const user = await User.findOne({ resetToken: token });
  return user;
}

async function updateUserPassword(user: any, password: string) {
  await user.updateOne({ password });
}
