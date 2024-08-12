import mongoose, { Document, Model, Schema } from 'mongoose';

export interface IUser {
  _id: string;
  name: string;
  email: string;
  password: string;
  phoneNumber: string;
  profilePicture: string;
  rideHistory: IRide[];
  createdAt: Date;
  updatedAt: Date;
}

const userSchema = new Schema({
  name: {
    type: String,
    required: true,
  },
  email: {
    type: String,
    required: true,
    unique: true,
  },
  password: {
    type: String,
    required: true,
  },
  phoneNumber: {
    type: String,
    required: true,
  },
  profilePicture: {
    type: String,
  },
  rideHistory: [
    {
      type: Schema.Types.ObjectId,
      ref: 'Ride',
    },
  ],
  createdAt: {
    type: Date,
    default: Date.now,
  },
  updatedAt: {
    type: Date,
    default: Date.now,
  },
});

const User: Model<IUser> = mongoose.model('User', userSchema);

export default User;
