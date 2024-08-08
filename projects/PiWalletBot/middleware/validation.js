import { check } from 'express-validator';

export function validateRequest(req, res, next) {
  const errors = check(req);
  if (errors.length > 0) {
    return res.status(400).json({ error: 'Invalid request' });
  }
  next();
}
