import { NextFunction, Request, Response } from 'express';
import { RateLimiter } from '../utils/rateLimiter';

const rateLimiter = new RateLimiter();

export function rateLimiterMiddleware(req: Request, res: Response, next: NextFunction) {
  const ip = req.ip;
  const limit = rateLimiter.getLimit(ip);
  if (limit.exceeded) {
    return res.status(429).json({ error: 'Rate limit exceeded' });
  }
  next();
}
