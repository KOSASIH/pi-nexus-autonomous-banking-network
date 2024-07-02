import jwt from 'jsonwebtoken';

const authMiddleware = (req, res, next) => {
  const token = req.headers['x-access-token'] || req.headers['authorization'];
  if (!token) {
    return res.status(401).send({ error: 'Unauthorized' });
  }
  jwt.verify(token, process.env.SECRET_KEY, (err, decoded) => {
    if (err) {
      return res.status(401).send({ error: 'Invalid token' });
    }
    req.user = decoded;
    next();
  });
};

export default authMiddleware;
