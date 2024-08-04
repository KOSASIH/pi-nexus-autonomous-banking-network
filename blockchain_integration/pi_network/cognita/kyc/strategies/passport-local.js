import { Strategy as LocalStrategy } from 'passport-local';
import { PassportStrategy } from '@nestjs/passport';
import { Injectable } from '@nestjs/common';

@Injectable()
export class LocalStrategy extends PassportStrategy(LocalStrategy) {
  constructor() {
    super({
      usernameField: 'email',
      passwordField: 'password',
    });
  }

  async validate(email: string, password: string) {
    // Implement local authentication logic here
    return { id: '1', email };
  }
}
