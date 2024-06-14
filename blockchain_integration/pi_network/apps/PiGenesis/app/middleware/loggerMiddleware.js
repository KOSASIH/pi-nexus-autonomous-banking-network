import { NextFunction, Request, Response } from 'express';

export function loggerMiddleware(req: Request, res: Response, next: NextFunction) {
  console.log(`Request: ${req.method} ${req.url}`);
  console.log(`Request Body: ${JSON.stringify(req.body)}`);
  next();
}

export function loggerResponseMiddleware(req: Request, res: Response, next: NextFunction) {
  console.log(`Response: ${res.statusCode} ${res.statusMessage}`);
  console.log(`Response Body: ${JSON.stringify(res.body)}`);
  next();
}
