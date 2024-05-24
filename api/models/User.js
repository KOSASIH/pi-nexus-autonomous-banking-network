import { model, Schema } from 'mongoose';
import bcrypt from 'bcrypt';
import jwt from 'jsonwebtoken';
import { v4 as uuidv4 } from 'uuid';

const userSchema = new Schema({
  _id: { type: String, default: uuidv4 },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  name: { type: String, required: true },
  role: { type: String, enum: ['user', 'admin'], default: 'user' },
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now }
});

userSchema.pre('save', async function(next) {
  const user = this;
  if (user.isModified('password')) {
    user.password = await bcrypt.hash(user.password, 10);
  }
  next();
});

userSchema.methods.generateToken = function() {
  const user = this;
  const token = jwt.sign({ _id: user._id, email: user.email }, process.env.SECRET_KEY, {
    expiresIn: '1h'
  });
  return token;
};

export default model('User', userSchema);
