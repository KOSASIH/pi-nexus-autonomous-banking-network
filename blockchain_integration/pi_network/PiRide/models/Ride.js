import mongoose, { Document, Model, Schema } from 'mongoose';

export interface IRide {
  _id: string;
  userId: string;
  pickupLocation: {
    latitude: number;
    longitude: number;
  };
  dropoffLocation: {
    latitude: number;
    longitude: number;
  };
  rideDate: Date;
  rideTime: string;
  rideType: string;
  seatsAvailable: number;
  price: number;
  status: string;
  createdAt: Date;
  updatedAt: Date;
}

const rideSchema = new Schema({
  userId: {
    type: Schema.Types.ObjectId,
    ref: 'User',
    required: true,
  },
  pickupLocation: {
    latitude: {
      type: Number,
      required: true,
    },
    longitude: {
      type: Number,
      required: true,
    },
  },
  dropoffLocation: {
    latitude: {
      type: Number,
      required: true,
    },
    longitude: {
      type: Number,
      required: true,
    },
  },
  rideDate: {
    type: Date,
    required: true,
  },
  rideTime: {
    type: String,
    required: true,
  },
  rideType: {
    type: String,
    required: true,
  },
  seatsAvailable: {
    type: Number,
    required: true,
  },
  price: {
    type: Number,
    required: true,
  },
  status: {
    type: String,
    required: true,
    enum: ['pending', 'accepted', 'rejected', 'completed'],
    default: 'pending',
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
  updatedAt: {
    type: Date,
    default: Date.now,
  },
});

rideSchema.index({ pickupLocation: '2dsphere' });
rideSchema.index({ dropoffLocation: '2dsphere' });

const Ride: Model<IRide> = mongoose.model('Ride', rideSchema);

export default Ride;
