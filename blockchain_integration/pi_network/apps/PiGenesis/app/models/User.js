import mongoose, { Document, Model, Schema } from 'ongoose';

interface User {
  username: string;
  password: string;
  email: string;
  address: string;
}

const userSchema = new Schema<User>({
  username: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  address: { type: String, required: true },
});

const User: Model<User> = mongoose.model('User', userSchema);

export default User;
