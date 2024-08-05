import { Injectable } from '@nestjs/common';
import { CanActivate, ExecutionContext, ForbiddenException } from '@nestjs/common';
import { Reflector } from '@nestjs/core';

@Injectable()
export class RolesGuard implements CanActivate {
  constructor(private reflector: Reflector) {}

  async canActivate(context: ExecutionContext) {
    const roles = this.reflector.get<string[]>('roles', context.getHandler());
    if (!roles) {
      return true;
    }
    const request = context.switchToHttp().getRequest();
    const user = request.user;
    if (!user || !user.roles) {
      throw new ForbiddenException('Forbidden');
    }
    const hasRole = () => user.roles.includes(roles[0]);
    if (!hasRole()) {
      throw new ForbiddenException('Forbidden');
    }
    return true;
  }
}
