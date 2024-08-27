export function errorHandler(err, req, res, next) {
  console.error(err);
  res.status(500).json({ error: 'Internal Server Error' });
}

export function validateRequest(req, res, next) {
  // implement request validation logic here
  next();
}
