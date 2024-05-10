const jwt = require('jsonwebtoken')
const User = require('../models/user')

const authenticate = async (req, res, next) => {
  const token = req.header('Authorization')?.split(' ')[1]

  if (!token) {
    return res.status(401).json({ message: 'Access Denied' })
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET)
    req.user = await User.findById(decoded.user_id)
    next()
  } catch (error) {
    res.status(400).json({ message: 'Invalid Token' })
  }
}

module.exports = authenticate
