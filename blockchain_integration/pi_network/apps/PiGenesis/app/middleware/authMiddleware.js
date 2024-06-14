import { NextFunction, Request, Response } from 'express';
import { authenticate, authorize } from '../utils/authUtils';

export function authenticateMiddleware(req: Request, res: Response, next: NextFunction) {
  const { username, password } = req.body;
  const user = authenticate(username, password);
  if (!user) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }
  req.user = user;
  next();
}

export function authorizeMiddleware(req: Request, res: Response, next: NextFunction) {
  const user = req.user;
  const token = authorize(user);
  if (!token) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  req.token = token;
  next();
}
