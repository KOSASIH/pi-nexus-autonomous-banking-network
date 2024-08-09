import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { User } from '../models/user.model';
import { Role } from '../enums/role.enum';
import { PaginatedResult } from '../interfaces/paginated-result.interface';

@Injectable()
export class UserService {
  constructor(@InjectModel('User') private readonly userModel: Model<User>) {}

  async findAll(page: number, limit: number): Promise<PaginatedResult<User>> {
    const users = await this.userModel
      .find()
      .skip((page - 1) * limit)
      .limit(limit)
      .populate('roles');
    const count = await this.userModel.countDocuments();
    return { data: users, count, page, limit };
  }

  async findById(id: string): Promise<User> {
    return this.userModel.findById(id).populate('roles');
  }

  async findByEmail(email: string): Promise<User> {
    return this.userModel.findOne({ email });
  }

  async create(user: User): Promise<User> {
    return this.userModel.create(user);
  }

  async update(id: string, user: User): Promise<User> {
    return this.userModel.findByIdAndUpdate(id, user, { new: true });
  }

  async delete(id: string): Promise<void> {
    await this.userModel.findByIdAndRemove(id);
  }

  async addRole(userId: string, role: Role): Promise<User> {
    return this.userModel.findByIdAndUpdate(userId, { $push: { roles: role } }, { new: true });
  }

  async removeRole(userId: string, role: Role): Promise<User> {
    return this.userModel.findByIdAndUpdate(userId, { $pull: { roles: role } }, { new: true });
  }
}
