import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { User } from '../models/user.model';
import { JwtService } from '@nestjs/jwt';
import { ConfigService } from '@nestjs/config';
import * as bcrypt from 'bcrypt';
import { Role } from '../enums/role.enum';
import { AccessToken } from '../interfaces/access-token.interface';
import { RefreshToken } from '../interfaces/refresh-token.interface';

@Injectable()
export class AuthService {
  constructor(
    @InjectModel('User') private readonly userModel: Model<User>,
    private readonly jwtService: JwtService,
    private readonly configService: ConfigService,
  ) {}

  async register(user: User): Promise<User> {
    const hashedPassword = await bcrypt.hash(user.password, 10);
    user.password = hashedPassword;
    return this.userModel.create(user);
  }

  async login(email: string, password: string): Promise<AccessToken> {
    const user = await this.userModel.findOne({ email });
    if (!user) {
      throw new Error('Invalid email or password');
    }
    const isValid = await bcrypt.compare(password, user.password);
    if (!isValid) {
      throw new Error('Invalid email or password');
    }
    const accessToken = this.generateAccessToken(user);
    const refreshToken = this.generateRefreshToken(user);
    return { accessToken, refreshToken };
  }

  async refreshToken(refreshToken: string): Promise<AccessToken> {
    const user = await this.userModel.findOne({ refreshToken });
    if (!user) {
      throw new Error('Invalid refresh token');
    }
    const accessToken = this.generateAccessToken(user);
    return { accessToken };
  }

  async logout(refreshToken: string): Promise<void> {
    await this.userModel.updateOne({ refreshToken }, { $set: { refreshToken: null } });
  }

  private generateAccessToken(user: User): string {
    const payload = { sub: user.id, email: user.email, roles: user.roles };
    return this.jwtService.sign(payload, {
      secret: this.configService.get('JWT_SECRET'),
      expiresIn: '1h',
    });
  }

  private generateRefreshToken(user: User): string {
    const payload = { sub: user.id, email: user.email };
    return this.jwtService.sign(payload, {
      secret: this.configService.get('JWT_SECRET'),
      expiresIn: '7d',
    });
  }
    }
