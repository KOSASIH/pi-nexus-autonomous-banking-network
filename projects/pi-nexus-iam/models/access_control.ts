import { Document, Model, model, Schema } from 'mongoose';
import { Role } from './role.enum';
import { Permission } from './permission.enum';

export interface AccessControl {
  _id: string;
  role: Role;
  permissions: Permission[];
  createdAt: Date;
  updatedAt: Date;
}

const accessControlSchema = new Schema<AccessControl>({
  role: { type: String, enum: Role, required: true },
  permissions: [{ type: String, enum: Permission, required: true }],
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now },
});

const AccessControl: Model<AccessControl> = model('AccessControl', accessControlSchema);

export default AccessControl;
