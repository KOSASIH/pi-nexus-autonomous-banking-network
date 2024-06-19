const jwt = require('jsonwebtoken');
const OAuth2 = require('oauth2');

const secretKey = 'your_secret_key_here';
const oauth2 = new OAuth2('your_client_id', 'your_client_secret', 'https://your_oauth2_server.com/token');

async function authenticateUser(token) {
  try {
    const decoded = jwt.verify(token, secretKey);
    const userId = decoded.sub;
    const user = await getUserFromDatabase(userId);
    return user;
  } catch (error) {
    throw new Error('Invalid token');
  }
}

async function generateToken(user) {
  const token = jwt.sign({ sub: user.id, name: user.name, email: user.email }, secretKey, {
    expiresIn: '1h',
  });
  return token;
}

async function authorizeRequest(req, res, next) {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.status(401).send({ error: 'Unauthorized' });
  }

  try {
    const user = await authenticateUser(token);
    req.user = user;
    next();
  } catch (error) {
    return res.status(401).send({ error: 'Invalid token' });
  }
}

module.exports = {
  authenticateUser,
  generateToken,
  authorizeRequest,
};
