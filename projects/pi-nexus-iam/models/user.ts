import { Document, Model, model, Schema } from 'mongoose';
import bcrypt from 'bcrypt';
import jwt from 'jsonwebtoken';
import { Role } from './role.enum';

export interface User {
  _id: string;
  username: string;
  email: string;
  password: string;
  roles: Role[];
  createdAt: Date;
  updatedAt: Date;
}

export interface UserDocument extends User, Document {
  comparePassword(password: string): Promise<boolean>;
  generateToken(): string;
}

const userSchema = new Schema<UserDocument>({
  username: { type: String, required: true, unique: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  roles: [{ type: String, enum: Role, required: true }],
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now },
});

userSchema.pre('save', async function(next) {
  const user = this;
  if (user.isModified('password')) {
    user.password = await bcrypt.hash(user.password, 10);
  }
  next();
});

userSchema.methods.comparePassword = async function(password: string) {
  const user = this;
  return await bcrypt.compare(password, user.password);
};

userSchema.methods.generateToken = function() {
  const user = this;
  const token = jwt.sign({ userId: user._id }, process.env.SECRET_KEY, {
    expiresIn: '1h',
  });
  return token;
};

const User: Model<UserDocument> = model('User', userSchema);

export default User;
