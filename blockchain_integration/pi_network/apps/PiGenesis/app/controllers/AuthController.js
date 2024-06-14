import { NextFunction, Request, Response } from 'express';
import { authenticate, authorize } from '../middleware/authMiddleware';
import { BlockchainService } from '../blockchain-integration/BlockchainService';

class AuthController {
  async login(req: Request, res: Response, next: NextFunction) {
    const { username, password } = req.body;
    const user = await authenticate(username, password);
    if (!user) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }
    const token = await authorize(user);
    res.json({ token });
  }

  async register(req: Request, res: Response, next: NextFunction) {
    const { username, password, email } = req.body;
    const user = await registerUser(username, password, email);
    if (!user) {
      return res.status(400).json({ error: 'Failed to register user' });
    }
    res.json({ message: 'User registered successfully' });
  }

  async logout(req: Request, res: Response, next: NextFunction) {
    req.logout();
    res.json({ message: 'Logged out successfully' });
  }
}

export default AuthController;
